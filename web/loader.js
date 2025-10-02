import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Me.Loader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.category !== "Me/loader") return;

        const origNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origNodeCreated) origNodeCreated.apply(this, arguments);

            const vaeFileWidget = this.widgets.find(w => w.name === "vae_file");
            const latentFileWidget = this.widgets.find(w => w.name === "latent_file");

            if (!vaeFileWidget || !latentFileWidget) return;

            // 开关状态
            this._toggleState = {
                vae: vaeFileWidget.value !== "None",
                latent: latentFileWidget.value !== "None"
            };

            // 创建开关UI
            this._toggles = [
                { label: "VAE", key: "vae", widget: vaeFileWidget },
                { label: "Latent", key: "latent", widget: latentFileWidget }
            ];

            // 获取文件列表的函数
            const getFileList = (widget) => {
                return widget.options.values || [];
            };

            // 设置widget值的函数
            const setWidgetValue = (widget, enabled) => {
                if (!enabled) {
                    widget.value = "None";
                    widget.disabled = true;
                } else {
                    const files = getFileList(widget);
                    if (files && files.length > 0 && files[0] !== "None") {
                        widget.value = files[0];
                    } else if (files && files.length > 1) {
                        widget.value = files[1]; // 跳过"None"
                    } else {
                        widget.value = "None";
                    }
                    widget.disabled = false;
                }
            };

            const toggleW = 36, toggleH = 18, spacing = 100;
            const origDraw = this.onDrawForeground;

            this.onDrawForeground = function (ctx) {
                if (origDraw) origDraw.apply(this, arguments);
                let y = 19;
                this._toggleAreas = [];

                this._toggles.forEach((t, i) => {
                    const x = 20 + i * spacing;

                    // label
                    ctx.fillStyle = "#333";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "middle";
                    ctx.fillText(t.label, x, y + toggleH / 2);

                    const toggleX = x + 40;
                    const isOn = this._toggleState[t.key];

                    // 背景
                    ctx.fillStyle = isOn ? "#4CAF50" : "#ccc";
                    ctx.beginPath();
                    ctx.roundRect(toggleX, y, toggleW, toggleH, 9);
                    ctx.fill();

                    // 圆点
                    ctx.fillStyle = "#fff";
                    const circleX = isOn ? toggleX + toggleW - 9 : toggleX + 9;
                    ctx.beginPath();
                    ctx.arc(circleX, y + toggleH / 2, 7, 0, Math.PI * 2);
                    ctx.fill();

                    this._toggleAreas.push({ x: toggleX, y, w: toggleW, h: toggleH, key: t.key, widget: t.widget });
                });
            };

            const origMouse = this.onMouseDown;
            this.onMouseDown = function (e, pos) {
                if (origMouse) origMouse.apply(this, arguments);
                if (!this._toggleAreas) return;

                for (const area of this._toggleAreas) {
                    if (pos[0] >= area.x && pos[0] <= area.x + area.w &&
                        pos[1] >= area.y && pos[1] <= area.y + area.h) {
                        const key = area.key;
                        const newState = !this._toggleState[key];
                        this._toggleState[key] = newState;

                        // 切换状态时设置文件选择器
                        setWidgetValue(area.widget, newState);

                        this.setDirtyCanvas(true, true);
                        return true;
                    }
                }
            };

            // 初始设置widget状态
            this._toggles.forEach(t => {
                setWidgetValue(t.widget, this._toggleState[t.key]);
            });
        };
    }
});