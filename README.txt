RK3588 楼宇电动车违规入内识别与本地报警系统重点代码说明

1. main_web_rule_editor_performance.py
最终主程序，包含摄像头/视频输入、RKNN模型推理、ROI电子围栏、入口线判断、连续三帧确认、LED报警、截图留证、SQLite日志、Web页面和性能显示。

2. configs/dorm_gate.yaml
实时摄像头模式配置文件。

3. configs/dorm_gate_video.yaml
本地视频演示模式配置文件。

4. config/roi_config.json
Web页面保存的电子围栏和入口线坐标。

5. system_service/ev-camera.service
Linux systemd 后台自启动服务文件，用于开发板开机后自动启动检测程序。

6. web_autostart/
开发板外接显示器时自动打开网页的桌面自启动文件和脚本。

说明：
本压缩包只包含最终重点代码和必要配置文件。
不包含RKNN模型文件、测试视频、报警截图、运行日志和数据库文件。
