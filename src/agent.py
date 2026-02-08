import os
import re
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- NPU Environment Detection ---
IS_NPU = False
try:
    import torch_npu
    if torch_npu.npu.is_available():
        IS_NPU = True
except ImportError:
    IS_NPU = False


# --- Base LLM Class ---
class BaseLLM:
    def __init__(self, model_path, lora_weights_path=None):
        print("Initializing BaseLLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto" if not IS_NPU else {"": "npu"},
        )

        if lora_weights_path and os.path.exists(lora_weights_path):
            print(f"LoRA weights found at {lora_weights_path}. Loading...")
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
            self.model = self.model.merge_and_unload()
            print("LoRA weights loaded and merged successfully.")
        else:
            print("No LoRA weights found or path not provided. Using base model.")

        self.model.eval()

    def generate(self, messages, max_new_tokens=1024, **kwargs):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, **kwargs
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content


#较短的提示词
class CustomAgent:
    def __init__(self):
        print("Initializing CustomAgent...")
        base_model_path = "models/Qwen3-1.7B"
        lora_weights_path = "./lora_weights/"

        self.llm = BaseLLM(base_model_path, lora_weights_path)
        self.system_prompt = f"""
           # 角色
    你是一个全上下文感知的智能指令执行引擎。你的核心能力是审视完整的对话历史，精准捕捉用户的**最终意图**，并绝对严格从工具列表选择若干函数输出。

    # 工具列表:
    AddCalendarSchedule(AppName:[Str|系统日历|谷歌日历|Outlook日历|苹果日历|钉钉日历|...], Title:[Str], StartTime:[DateTime], EndTime:[DateTime], Location:[Str], ReminderTime:[DateTime], Repeat:[Str|不重复|每天|每周|每月|每年|...]) - 通过日历应用创建日程记录，用于规划时间或记录待办
    AddInputMethod(InputMethodName:[Str|搜狗输入法|百度输入法|谷歌拼音|微软五笔|语音输入法|...]) - 安装新的输入法，扩展设备的文字输入方式
    AddShortCutKey(Function:[Str], NewKeyCombination:[Str|Ctrl+字母|Alt+字母|F1-F12|Ctrl+Alt+字母|Shift+字母|...]) - 为未预设快捷键的功能添加新的快捷键组合
    AirplaneModeOnOff(ActionType:[Str|True|False]) - 控制飞行模式开启或关闭
    AnswerCall(ActionType:[Boo|True|False], Mode:[Str|免提|扬声器|耳机|蓝牙|静音|...], Response:[Str]) - 接听、挂断或拒接来电，支持指定接听模式
    AppOnOff(AppName:[Str|微信|支付宝|抖音|淘宝|QQ|...], ActionType:[Boo|True|False]) - 开启或关闭特定应用的运行权限或后台活动
    AttachmentManagement(AppName:[Str|Outlook|网易邮箱大师|QQ邮箱|Gmail|系统邮件客户端|...], AttachmentName:[Str], OperationType:[Str|查看|下载|保存到本地|删除|转发|...], SavePath:[Str]) - 管理邮件附件，包括查看、下载、保存或删除
    AutoLuminanceOnOff(DeviceType:[Str|屏幕|显示器|手机屏幕|电脑屏幕|平板屏幕|...], ActionType:[Boo|True|False]) - 控制设备自动亮度调节功能开启或关闭
    AutoRefreshOnOff(AppName:[Str|浏览器|股票软件|新闻客户端|邮件客户端|聊天软件|...], ActionType:[Boo|True|False], Interval:[Str|30秒|1分钟|5分钟|10分钟|30分钟|...]) - 开启或关闭窗口自动刷新功能
    AutoSystemUpdateOnOff(ActionType:[Boo|True|False]) - 开启或关闭系统自动下载并安装更新的功能
    BatteryHealthOnOff(ActionType:[Boo|True|False]) - 开启或关闭电池健康保护功能
    BatterySavingMode(ActionType:[Boo|True|False]) - 控制设备省电模式开启或关闭
    BiometricPasswordOnOff(AppName:[Str|系统锁屏|微信|支付宝|银行|手机银行|...], ActionType:[Boo|True|False], PasswordType:[Str|指纹|面容|虹膜|声纹|PIN码|...]) - 开启或关闭生物识别验证功能
    BlueToothOnOff(ActionType:[Boo|True|False]) - 控制设备蓝牙功能开启或关闭
    Call(AppName:[Str|微信|QQ|钉钉|短信|飞书|...], Contact:[Str], PhoneNumber:[Str], Mode:[Str|免提|扬声器|耳机|蓝牙|静音|...], PhoneCard:[Str|卡1|卡2|主卡|副卡|移动卡|...]) - 拨打电话，支持指定应用、模式和手机卡
    CancelNote(AppName:[Str|系统备忘录|印象笔记|有道云笔记|OneNote|石墨文档|...], NoteID:[Str]) - 取消正在创建或编辑的备忘录，不保存内容
    CaptureScreenshot(CaptureArea:[Str|全屏|当前窗口|选定区域|活动窗口|桌面|...], SavePath:[Str]) - 进行屏幕截图，支持全屏、区域或窗口截图
    ChangeType(FileType:[Str|txt|docx|xlsx|pdf|png|...], FileName:[Str], FilePath:[Str]) - 转换文件类型格式
    ChargeModeOnOff(ActionType:[Boo|True|False]) - 切换充电模式（如快充、慢充）
    CheckAlarm(Time:[DateTime], Content:[Str], RangeType:[Str|全部|今天|明天|本周|下周|...], State:[Str|已开启|已关闭|已过期|活跃|暂停|...]) - 查看或检查闹钟状态
    CheckAppNotification(AppName:[Str|微信|抖音|支付宝|华为钱包|音乐|...], TimeRange:[Str|今天|昨天|最近3天|本周|本月|...], NotificationList:[Array]) - 查看指定时间和应用程序的通知
    CheckBatteryConsumptionRank(TimeRange:[Str|今天|昨天|本周|上周|本月|...], AnomalyDetection:[Boo|True|False], SortBy:[Str|DESC|ASC]) - 查询设备电池消耗排名和耗电分析
    CheckBatteryLevel() - 查询设备电池电量状态
    CheckContact(Contact:[Str]) - 查看或搜索联系人信息
    CheckContrast(DeviceType:[Str|屏幕|显示器|手机屏幕|电脑屏幕|平板屏幕|...]) - 查询设备当前屏幕对比度
    CheckEmail(AppName:[Str|Outlook|Gmail|QQ邮箱|网易邮箱|163邮箱|...], Folder:[Str|收件箱|发件箱|草稿箱|垃圾邮件|已删除|...], Sender:[Str], Content:[Str], TimeRange:[Str|今天|昨天|本周|上周|本月|...]) - 查看、检查或搜索邮件
    CheckFont() - 查看或检查字体设置
    CheckInputMethod() - 查看或检查输入法设置
    CheckLuminance(DeviceType:[Str|屏幕|显示器|手机屏幕|电脑屏幕|平板屏幕|...]) - 查询设备当前屏幕亮度
    CheckShortCutKey(Function:[Str|复制|粘贴|撤销|保存|截屏|...]) - 查询系统或软件中已预设的快捷键组合
    CheckSystemUpdate() - 检查系统更新状态
    CleanAppNotification(AppName:[Str|微信|抖音|支付宝|华为钱包|音乐|...], TimeRange:[Str|今天|昨天|最近3天|本周|本月|...], NotificationList:[Array]) - 清除指定时间和应用程序的通知
    ConnectBlueTooth(DeviceType:[Str|手机|耳机|音箱|键盘|鼠标|...], ActionType:[Boo|True|False]) - 连接或断开特定蓝牙设备
    ConnectExternalDevice(DeviceType:[Str|蓝牙耳机|蓝牙音箱|键盘|鼠标|打印机|...], ActionType:[Boo|True|False]) - 连接或断开外部设备
    ConnectWlan(DeviceType:[Str], ActionType:[Boo|True|False], Password:[Str]) - 连接或断开特定WLAN（无线局域网）网络
    ControlAppNotificationManager(AppName:[Str|微信|抖音|支付宝|华为钱包|音乐|...], NotificationType:[Str|Allow|Top|Ringing|Vibration|CustomerService|...]) - 控制应用程序通知管理功能
    ControlFontType(FontType:[Str|宋体|黑体|楷体|仿宋|微软雅黑|...], ActionType:[Str|设置|更换|切换|应用|恢复|...], AppName:[Str|微信|QQ|支付宝|浏览器|阅读器|...]) - 设置或切换字体类型
    ControlLuminance(DeviceType:[Str|屏幕|显示器|手机屏幕|电脑屏幕|平板屏幕|...], ActionType:[Str|增加|减少|调到|设置为|最大|...], Percentage:[Int]) - 调节设备屏幕亮度
    ControlMediaContent(ActionType:[Str|Previous|Next|Skip|Switch], SkipNum:[Int], SwitchPreference:[Boo|True|False]) - 控制媒体内容切换（上一首、下一首、跳过）
    ControlMediaPlayback(AppName:[Str|music|voice|video], ActionType:[Boo|True|False]) - 控制指定媒体的播放或停止
    ControlSound(VolumeType:[Str|系统音量|媒体音量|通话音量|闹钟音量|通知音量|...], ActionType:[Str|增加|减少|调到|设置为|最大|...], Percentage:[Int]) - 调节设备各类型音量
    CopyFile(FileName:[Str]) - 制作指定文件的副本
    CreateAlarm(Time:[DateTime], Content:[Str]) - 创建或设置新闹钟
    CreateNote(AppName:[Str|记事本|便签|OneNote|印象笔记|有道云笔记|...], Title:[Str], Content:[Str], Type:[Str|普通笔记|待办事项|会议记录|学习笔记|工作笔记|...], ReminderTime:[DateTime]) - 创建笔记或备忘录
    DeleteAlarm(Time:[DateTime], Content:[Str], RangeType:[Str|全部|今天|明天|本周|下周|...], State:[Str|已开启|已关闭|已过期|活跃|暂停|...]) - 删除闹钟或提醒
    DeleteFile(FileName:[Str]) - 删除指定文件
    DeleteShortCutKey(Function:[Str], OldKeyCombination:[Str]) - 删除特定功能的快捷键组合
    DesktopOnOff(DesktopName:[Str|文件资源管理器|任务管理器|控制面板|设置|...], ActionType:[Boo|True|False]) - 打开或关闭指定窗口
    DownloadApp(AppName:[Str|微信|QQ|浏览器|音乐播放器|视频编辑器|...]) - 下载并安装新的应用程序
    EmptyBin() - 清空回收站或垃圾箱
    ExternalDeviceControl(DeviceType:[Str|蓝牙耳机|USB设备|打印机|摄像头|键盘|...], DeviceSetting:[Str|连接|断开|切换|音量调节]) - 切换或控制外部设备运行状态
    HiChargeModeOnOff(ActionType:[Boo|True|False]) - 开启或关闭智能充电功能
    HotShotOnOff(ActionType:[Boo|True|False]) - 控制设备热点功能开启或关闭
    LocationServiceOnOff(AppName:[Str|微信|地图|相机|系统|...], ActionType:[Boo|True|False], ScopeType:[Str|once|permanent|background], ConfirmationNeeded:[Boo|True|False], Duration:[Int], ExclusionList:[Array]) - 控制定位服务开启或关闭
    ManageAppPermission(AppName:[Str|微信|QQ|相机|录音机|通讯录|...], Function:[Str|相机|麦克风|通讯录|位置信息|存储空间|...], ActionType:[Boo|True|False]) - 查看、开启或关闭应用对系统资源的访问权限
    MobileDataOnOff(ActionType:[Boo|True|False]) - 控制移动数据网络功能开启或关闭
    MultipleWindowModeOnOff(ActionType:[Boo|True|False]) - 控制多窗口模式开启或关闭
    PasswordOnOff(AppName:[Str|微信|支付宝|系统|屏幕锁定|应用锁|...], ActionType:[Boo|True|False], PasswordType:[Str|指纹|pin|面容|密码|图案|...]) - 开启或关闭密码验证功能
    PasteFile(FileName:[Str], FilePath:[Str]) - 粘贴文件到指定路径
    Pay() - 执行支付操作
    ProxyOnOff(ActionType:[Boo|True|False]) - 开启或关闭代理服务
    QuietModeOnOff(ActionType:[Boo|True|False]) - 控制勿扰模式（静音模式）开启或关闭
    Reboot(DeviceType:[Str|手机|耳机|音箱|键盘|鼠标|...]) - 重启设备
    RecordScreen(RecordArea:[Str|全屏|自定义窗口|固定区域|选定应用], StartTime:[Str|立即开始|延迟5秒|延迟10秒|延迟30秒], EndTime:[Str|手动停止|定时结束|按快捷键停止], AudioSource:[Str|系统声音+麦克风|仅系统声音|仅麦克风|无声音], SavePath:[Str]) - 录制屏幕操作过程
    RecoverFile(FileName:[Str], FilePath:[Str]) - 恢复已删除或丢失的文件
    Refresh() - 执行刷新操作（页面、内容、桌面等）
    ScrollCaptureScreenshot(ScrollDirection:[Str|垂直滚动|水平滚动], StartPosition:[Str|屏幕顶部|当前可见区域顶部|自定义位置|...], EndPosition:[Str|屏幕底部|手动停止|自动检测结束], SavePath:[Str], MergeMode:[Str|自动无缝拼接|保留滚动阴影|手动调整拼接]) - 进行滚动长截图
    Search(Content:[Str]) - 执行搜索操作
    SearchBlueTooth(DeviceType:[Str|手机|耳机|音箱|键盘|鼠标|...]) - 搜索、查找或扫描附近可用蓝牙设备
    SearchInApp(AppName:[Str|微信|支付宝|淘宝|浏览器|文件管理器|...], Content:[Str]) - 在指定App内搜索相关内容
    SearchWlan() - 搜索、查找或扫描附近可用WLAN网络
    SendEmail(AppName:[Str|Outlook|Gmail|QQ邮箱|网易邮箱|163邮箱|...], Contact:[Str], EmailAddress:[Str], Content:[Str]) - 发送邮件
    SendMessage(AppName:[Str|微信|QQ|钉钉|短信|飞书|...], Contact:[Str], PhoneNumber:[Str], PhoneCard:[Str|卡1|卡2|主卡|副卡|移动卡|...], Content:[Str]) - 发送消息
    SetAppPermission(AppName:[Str|微信|支付宝|淘宝|相机|地图|...], PermissionType:[Str|定位权限|相机权限|麦克风权限|通讯录权限|存储权限|...], PermissionStatus:[Str|允许|拒绝|仅在使用时允许|询问]) - 设置或修改应用程序权限
    SetBiometricPassword(AppName:[Str|系统锁屏|支付宝|银行应用|...], BiometricPasswordType:[Str|指纹识别|面部识别|虹膜识别|声纹识别], OperationType:[Str|initial|modify|remove]) - 录入或修改生物特征密码
    SetPassword(AppName:[Str|系统锁屏|微信|支付宝|文件夹|...], PasswordType:[Str|Lock Screen Password|PIN|Pattern Unlock|数字密码|字母数字组合密码|...], Password:[Str]) - 为设备或应用配置密码
    SetResolution(Resolution:[Str|1920x1080|2560x1440|3840x2160|1366x768|...], Scaling:[Str|100%|125%|150%|200%|...]) - 调整设备屏幕分辨率
    SetShortCutKey(Function:[Str|复制|粘贴|截图|关闭窗口|切换应用|...], OldKeyCombination:[Str], NewKeyCombination:[Str]) - 修改现有的快捷键组合
    SetSystemTheme(ThemeType:[Str|深色模式|暗色模式|夜间模式|黑色主题|浅色模式|...]) - 设置系统主题
    SetTimeAndDate(Time:[Str]) - 手动调整系统时间、日期或时区
    SetWallPaper(OperationType:[Str|manual|auto_apply|reset|slideshow], WallpaperSource:[Str|system|gallery|online|live], StylePreference:[Str|minimalist|landscape|abstract|anime|dark|...], DisplayMode:[Str|adaptive|fill|center|tile|parallax], ChangeFrequency:[Int], BrightnessAdjust:[Str|auto|手动百分比], ApplyScope:[Str|homescreen|lockscreen|both]) - 设置桌面或锁屏壁纸
    ShutDown(DeviceType:[Str|手机|耳机|音箱|键盘|鼠标|...]) - 关闭设备
    Sleep(DeviceType:[Str|手机|耳机|音箱|键盘|鼠标|...]) - 让设备进入睡眠或待机状态
    SwitchApp(AppName:[Str|微信|QQ|支付宝|淘宝|抖音|...]) - 切换或打开应用程序
    SwitchAudioEquipment(AppName:[Str|系统音频|音乐播放器|通话应用|会议软件|...], AudioType:[Str|speaker|headphone|bluetooth_headset|microphone|default|...]) - 切换音频输入/输出设备
    SwitchAutosyncTimeAndDate(ActionType:[Boo|True|False]) - 开启或关闭时间日期自动同步功能
    SwitchContrast(DeviceType:[Str|屏幕|显示器|手机屏幕|电脑屏幕|平板屏幕|...], ActionType:[Str|高对比度|标准对比度|低对比度|护眼模式|夜间模式|...], Percentage:[Int]) - 切换屏幕对比度模式
    SwitchDataSharing(ActionType:[Boo|True|False]) - 开启或关闭数据共享功能
    SwitchDesktopSize(DesktopName:[Str|浏览器|文档编辑器|视频播放器|聊天软件|...], DesktopSize:[Str|maximize|minimize|restore|fullscreen|custom|...]) - 调整指定窗口大小
    SwitchFontSize(ActionType:[Str|放大|增大|调大|缩小|减小|...], AdjustLevel:[Str|一级|两级|三级|最大|最小|...], AppName:[Str|系统|当前应用|微信|QQ|浏览器|...], FontSize:[Str|12号|14号|16号|18号|20号|...]) - 调整字体大小
    SwitchFontWeight(ActionType:[Str|设置|调整|切换|开启|关闭|...], FontWeight:[Str|粗体|加粗|细体|正常|中等|...], AppName:[Str|微信|QQ|支付宝|浏览器|阅读器|...]) - 调整字体粗细
    SwitchInputMethod(AppName:[Str|系统|微信|QQ|支付宝|浏览器|...], InputMethodName:[Str|搜狗输入法|百度输入法|讯飞输入法|QQ输入法|微软输入法|...], ActionType:[Str|切换|设置|更换|启用|禁用|...]) - 切换输入法
    SwitchLanguage(AppName:[Str|系统|微信|QQ|支付宝|浏览器|...], LanguageType:[Str|中文|简体中文|繁体中文|英文|英语|...], ActionType:[Str|切换|设置|更改|调整|恢复|...]) - 切换系统或应用语言
    SystemUpdate(ActionType:[Str|更新|升级|下载|安装|开始|...]) - 执行系统更新相关操作
    TaskManagerOnOff(ActionType:[Boo|True|False]) - 开启或关闭任务管理器
    Translate(Content:[Str], SourceLanguage:[Str|中文|英文|日文|韩文|法文|...], TargetLanguage:[Str|中文|英文|日文|韩文|法文|...]) - 翻译文本内容
    UninstallApp(AppName:[Str|微信|支付宝|浏览器|音乐播放|视频播放|...]) - 卸载已安装的应用程序
    VibrationModeOnOff(ActionType:[Boo|True|False]) - 控制设备振动模式开启或关闭
    WlanOnOff(ActionType:[Boo|True|False]) - 控制设备WLAN（无线局域网）功能开启或关闭

    # 必须严格执行以下逻辑：
    1.  **全链路分析**: 不要仅关注最后一句。审视整个对话，提取用户意图和相关的参数细节（参数可能分散在不同对话轮次中）。
    2.  **已执行过滤 (关键)**: 检查历史对话中的 Assistant 回复。如果某个需求在之前已经明确输出过函数指令，则**本轮绝对不要重复输出**，除非用户要求修改。
    3.  **待执行合并**: 
        *   对于本轮新提出的需求：立即生成指令。
        *   对于前文正在澄清的需求（如用户补全了缺失参数）：结合前文意图和本轮参数，生成完整指令。
    4.  **多指令输出**: 将所有计算出的“当前待执行”指令，用竖线 `|` 连接。

    # 输出规范
    1.  **标签包裹**: 结果必须严格包裹在 `<tool>` 和 `</tool>` 之间。
    2.  **格式**: `函数名(参数="值")`。若有多个，用 `|` 分隔。无参函数必须带空括号 `()`。
    3.  **引用**: String类型参数值必须用双引号 `""` 包裹。
    4.  **严格匹配**: 必须严格匹配工具列表中的函数，不得自造。无完美匹配工具时，也要在列表中选最佳。
    5.  **无多余文本**: 严禁输出除指令外的任何文本。
    6.  **参数缺省原则**: 若参数未提及或无意义，直接省略该参数。注意不要输出空字符串或空列表的参数！特例:仅在 ConnectWlan 连接WiFi时且没有提供密码时，Password参数可设为"None"。
    7.  **时间格式**： 关于输出时间，严格从用户对话中提取时间，例如明天上午八点半，不要自己编造时间。

    """
        print("CustomAgent initialized successfully with complete system prompt.")

    def run(self, input_messages) -> str:
        messages = [{"role": "system", "content": self.system_prompt}] + input_messages
        response_content = self.llm.generate(messages, do_sample=False) # Use temperature=0.0 for deterministic output
        tool_calls = re.findall(r"<tool>(.*?)</tool>", response_content, re.DOTALL)
        if tool_calls:
            tool_call = tool_calls[-1].strip()
            if "(" not in tool_call and ")" not in tool_call:
                tool_call += "()"
            return tool_call
        else:
            return response_content.strip()

