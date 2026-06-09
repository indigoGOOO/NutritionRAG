"""Lightweight chat page served by FastAPI."""

CHAT_PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>膳食营养助手</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, "Microsoft YaHei", system-ui, sans-serif;
      background: #f5f7f4;
      color: #172018;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: stretch;
    }
    agent-chat {
      display: block;
      min-height: 100vh;
    }
  </style>
</head>
<body>
  <agent-chat></agent-chat>

  <script>
    class AgentChat extends HTMLElement {
      constructor() {
        super();
        this.messages = [];
        this.sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
        this.userId = localStorage.getItem("nutrition_user_id") || "default";
        localStorage.setItem("nutrition_user_id", this.userId);
        this.attachShadow({ mode: "open" });
      }

      connectedCallback() {
        this.render();
      }

      render() {
        this.shadowRoot.innerHTML = `
          <style>
            * { box-sizing: border-box; }
            .shell {
              min-height: 100vh;
              display: grid;
              grid-template-rows: auto 1fr auto;
              background: #f5f7f4;
            }
            header {
              padding: 18px 22px;
              border-bottom: 1px solid #dfe6dc;
              background: #ffffff;
            }
            h1 {
              margin: 0;
              font-size: 20px;
              line-height: 1.25;
              letter-spacing: 0;
            }
            .sub {
              margin-top: 4px;
              color: #5c6b5f;
              font-size: 13px;
            }
            main {
              overflow: auto;
              padding: 20px;
            }
            .messages {
              max-width: 920px;
              margin: 0 auto;
              display: grid;
              gap: 12px;
            }
            .msg {
              max-width: 78%;
              padding: 12px 14px;
              border-radius: 8px;
              line-height: 1.65;
              white-space: pre-wrap;
              word-break: break-word;
              border: 1px solid transparent;
            }
            .user {
              justify-self: end;
              background: #1f6f43;
              color: white;
            }
            .assistant {
              justify-self: start;
              background: white;
              border-color: #dfe6dc;
            }
            .meta {
              margin-top: 8px;
              color: #637067;
              font-size: 12px;
            }
            form {
              border-top: 1px solid #dfe6dc;
              background: #ffffff;
              padding: 14px;
            }
            .composer {
              max-width: 920px;
              margin: 0 auto;
              display: grid;
              grid-template-columns: 1fr auto;
              gap: 10px;
            }
            textarea {
              min-height: 48px;
              max-height: 160px;
              resize: vertical;
              border: 1px solid #cbd7ca;
              border-radius: 8px;
              padding: 12px;
              font: inherit;
              line-height: 1.45;
            }
            button {
              width: 88px;
              border: 0;
              border-radius: 8px;
              background: #1f6f43;
              color: white;
              font: inherit;
              font-weight: 600;
              cursor: pointer;
            }
            button:disabled {
              opacity: .55;
              cursor: not-allowed;
            }
            @media (max-width: 640px) {
              .msg { max-width: 94%; }
              .composer { grid-template-columns: 1fr; }
              button { width: 100%; height: 44px; }
            }
          </style>
          <section class="shell">
            <header>
              <h1>膳食营养助手</h1>
              <div class="sub">连接本地 FastAPI Agent 后端</div>
            </header>
            <main>
              <div class="messages">
                ${this.messages.length ? this.messages.map(m => this.messageTemplate(m)).join("") : `
                  <div class="msg assistant">你好，我可以帮你查询营养知识、分析饮食限制、结合你的画像给出膳食建议。</div>
                `}
              </div>
            </main>
            <form>
              <div class="composer">
                <textarea placeholder="输入你的问题，比如：我花生过敏，晚餐怎么吃？"></textarea>
                <button type="submit">发送</button>
              </div>
            </form>
          </section>
        `;

        this.shadowRoot.querySelector("form").addEventListener("submit", (event) => {
          event.preventDefault();
          this.send();
        });
      }

      messageTemplate(message) {
        const meta = message.intent ? `<div class="meta">intent: ${this.escape(message.intent)}</div>` : "";
        return `<div class="msg ${message.role}">${this.escape(message.content)}${meta}</div>`;
      }

      async send() {
        const textarea = this.shadowRoot.querySelector("textarea");
        const button = this.shadowRoot.querySelector("button");
        const text = textarea.value.trim();
        if (!text) return;

        this.messages.push({ role: "user", content: text });
        textarea.value = "";
        button.disabled = true;
        this.messages.push({ role: "assistant", content: "正在思考..." });
        this.render();

        try {
          const history = this.messages
            .filter(m => m.content !== "正在思考...")
            .slice(-12);
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message: text,
              session_id: this.sessionId,
              user_id: this.userId,
              history
            })
          });
          const data = await response.json();
          this.messages.pop();
          if (!response.ok) {
            this.messages.push({ role: "assistant", content: data.detail || "请求失败" });
          } else {
            this.messages.push({
              role: "assistant",
              content: data.answer || "没有生成回答。",
              intent: data.intent || ""
            });
          }
        } catch (error) {
          this.messages.pop();
          this.messages.push({ role: "assistant", content: `请求失败：${error}` });
        } finally {
          button.disabled = false;
          this.render();
        }
      }

      escape(value) {
        return String(value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;");
      }
    }

    customElements.define("agent-chat", AgentChat);
  </script>
</body>
</html>
"""
