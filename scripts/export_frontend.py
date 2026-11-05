import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.api.app_stream import INDEX_HTML

os.makedirs("frontend", exist_ok=True)

# Replace the hardcoded wsUrl to allow connecting to an external backend if deployed on Netlify
target_snippet = """        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${proto}//${window.location.host}/ws/s2s_live?events=true&format=wav`;"""

replacement_snippet = """        let customServer = (document.getElementById('serverUrlInput') && document.getElementById('serverUrlInput').value || '').trim();
        let wsUrl = '';
        if (customServer) {
          customServer = customServer.replace(/^http/, 'ws');
          if (!customServer.startsWith('ws://') && !customServer.startsWith('wss://')) {
            customServer = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + customServer;
          }
          customServer = customServer.replace(/\\/+$/, '');
          wsUrl = `${customServer}/ws/s2s_live?events=true&format=wav`;
        } else {
          const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
          wsUrl = `${proto}//${window.location.host}/ws/s2s_live?events=true&format=wav`;
        }"""

html = INDEX_HTML.replace(target_snippet, replacement_snippet)

# Add a config bar for Backend URL
card_target = '<div class="card">'
config_bar = """<div class="card" style="padding: 12px 18px; margin-bottom: 4px; display: flex; align-items: center; gap: 12px; background: rgba(19, 27, 46, 0.95); border: 1px solid #1e293b; border-radius: 12px;">
      <span style="font-size: 13px; font-weight: 600; color: #94a3b8; white-space: nowrap;">API Server URL:</span>
      <input id="serverUrlInput" type="text" placeholder="Để trống nếu chạy chung host (hoặc điền: https://your-backend.railway.app)" style="flex: 1; background: #090d16; border: 1px solid #334155; color: #f8fafc; padding: 7px 12px; border-radius: 6px; font-size: 13px; outline: none;" />
    </div>
    <div class="card">"""

html = html.replace(card_target, config_bar, 1)

# Route API fetch requests through customServer if set
html = html.replace('fetch("/api/v1/', 'fetch((document.getElementById("serverUrlInput") && document.getElementById("serverUrlInput").value.trim().replace(/\\/+$/, "") || "") + "/api/v1/')

with open("frontend/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Exported frontend/index.html successfully!")
