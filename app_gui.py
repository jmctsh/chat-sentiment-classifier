import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import BertTokenizer
from volcenginesdkarkruntime import Ark
from typing import Dict
import time
import threading

from models.model import BertMLPClassifier
from configs.config import MODEL_CONFIG

DOUBAO_API_KEY = "f14b5838-9d27-4587-aa06-e26eed9c57e1"
DOUBAO_MODEL = "doubao-seed-2-0-pro-260215"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_model.pt")

LABEL_MAP = {
    0: "善意",
    1: "辱骂", 
    2: "中性",
    3: "中性玩梗"
}

SYSTEM_PROMPT = """你是一个贴吧老哥，说话风格要接地气、有网感。
回复要求：
- 用词口语化，可以适当使用网络用语
- 回复要简短有力，不要长篇大论
- 可以使用一些贴吧常见的表情和语气词
- 保持真实感，像真人聊天一样"""

def log(msg):
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")

class ChatApp:
    def __init__(self):
        log("初始化模型...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log(f"使用设备: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
        
        self.model = BertMLPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_labels=MODEL_CONFIG['num_labels'],
            hidden_size=MODEL_CONFIG['hidden_size'],
            dropout_rate=MODEL_CONFIG['dropout_rate']
        ).to(self.device)
        
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            log("模型加载完成")
        else:
            raise FileNotFoundError(f"找不到模型: {CHECKPOINT_PATH}")
        
        self.model.eval()
        
        self.doubao_client = Ark(
            base_url=DOUBAO_BASE_URL,
            api_key=DOUBAO_API_KEY,
        )
        
        self.chat_history = []
        log("初始化完成!")
    
    def classify_text(self, text: str) -> Dict:
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        else:
            token_type_ids = torch.zeros((1, 128), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()
        
        all_scores = {
            LABEL_MAP[i]: round(probs[0, i].item(), 4)
            for i in range(len(LABEL_MAP))
        }
        
        return {
            'label': LABEL_MAP[pred_label],
            'label_id': pred_label,
            'confidence': confidence,
            'all_scores': all_scores
        }
    
    def get_doubao_response(self, user_text: str, classification: Dict) -> str:
        label_id = classification['label_id']
        
        if label_id == 0:
            prompt_suffix = "\n\n用户说的话是善意的，你要友善地回应，保持贴吧老哥的风格。"
            use_search = False
        elif label_id == 1:
            prompt_suffix = "\n\n用户在骂人/恶意攻击，你要用贴吧老哥的风格怼回去，不要客气，但要文明用语，不要真的骂脏话。"
            use_search = False
        elif label_id == 2:
            prompt_suffix = "\n\n用户说的是中性内容，正常回复即可，保持贴吧老哥的风格。"
            use_search = False
        else:
            prompt_suffix = "\n\n用户在玩梗/说网络流行语。如果你知道这个梗，就接梗回复；如果你不知道这个梗，请先搜索了解后再回复。"
            use_search = True
        
        messages = []
        for msg in self.chat_history:
            messages.append({"role": msg["role"], "content": str(msg["content"])})
        
        messages.append({"role": "user", "content": f"{SYSTEM_PROMPT}\n\n用户说：{user_text}{prompt_suffix}"})
        
        try:
            if use_search:
                tools = [{"type": "web_search", "max_keyword": 3}]
                response = self.doubao_client.responses.create(
                    model=DOUBAO_MODEL,
                    input=messages,
                    tools=tools,
                )
            else:
                response = self.doubao_client.responses.create(
                    model=DOUBAO_MODEL,
                    input=messages,
                )
            
            if hasattr(response, 'output'):
                assistant_reply = str(response.output)
            elif hasattr(response, 'choices') and response.choices:
                assistant_reply = str(response.choices[0].message.content)
            else:
                assistant_reply = str(response)
            
            self.chat_history.append({"role": "user", "content": str(user_text)})
            self.chat_history.append({"role": "assistant", "content": assistant_reply})
            
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            
            return assistant_reply
            
        except Exception as e:
            log(f"豆包API错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"豆包API调用出错: {str(e)}"
    
    def classify_only(self, user_input: str) -> tuple:
        if not user_input or not user_input.strip():
            return None, "请输入内容"
        
        classification = self.classify_text(user_input)
        
        scores_text = "各分类得分:\n"
        for lbl, score in classification['all_scores'].items():
            scores_text += f"  {lbl}: {score:.2%}\n"
        
        return classification, scores_text
    
    def get_reply(self, user_input: str, classification: Dict) -> str:
        return self.get_doubao_response(user_input, classification)
    
    def clear(self):
        self.chat_history = []
        return "对话历史已清空"


import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

class ChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("游戏对话情绪分类助手")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        self.app = ChatApp()
        
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(
            main_frame, 
            text="🎮 游戏对话情绪分类助手",
            font=('Microsoft YaHei', 16, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        input_frame = ttk.LabelFrame(main_frame, text="输入", padding="5")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame, 
            height=3, 
            font=('Microsoft YaHei', 11),
            wrap=tk.WORD
        )
        self.input_text.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.send_btn = ttk.Button(
            btn_frame, 
            text="发送", 
            command=self.send_message,
            width=15
        )
        self.send_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            btn_frame, 
            text="清空对话", 
            command=self.clear_chat,
            width=15
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(btn_frame, text="就绪", foreground='green')
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        output_frame = ttk.LabelFrame(result_frame, text="回复结果", padding="5")
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame, 
            height=15, 
            font=('Microsoft YaHei', 11),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scores_frame = ttk.LabelFrame(result_frame, text="分类详情", padding="5")
        scores_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        self.scores_text = scrolledtext.ScrolledText(
            scores_frame, 
            width=25, 
            height=15, 
            font=('Microsoft YaHei', 10),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.scores_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        legend_frame = ttk.LabelFrame(scores_frame, text="分类说明", padding="5")
        legend_frame.pack(fill=tk.X, padx=5, pady=5)
        
        legend_text = """🟢 善意 → 友善回应
🔴 辱骂 → 怼回去
⚪ 中性 → 正常回复
🟡 玩梗 → 接梗/搜索"""
        
        legend_label = ttk.Label(legend_frame, text=legend_text, font=('Microsoft YaHei', 9))
        legend_label.pack()
        
        self.input_text.bind('<Control-Return>', lambda e: self.send_message())
    
    def update_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state=tk.DISABLED)
    
    def update_scores(self, text):
        self.scores_text.config(state=tk.NORMAL)
        self.scores_text.delete("1.0", tk.END)
        self.scores_text.insert(tk.END, text)
        self.scores_text.config(state=tk.DISABLED)
    
    def send_message(self):
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return
        
        self.status_label.config(text="分类中...", foreground='orange')
        self.send_btn.config(state=tk.DISABLED)
        self.root.update()
        
        classification, scores_text = self.app.classify_only(user_input)
        
        if classification is None:
            self.update_output(scores_text)
            self.status_label.config(text="就绪", foreground='green')
            self.send_btn.config(state=tk.NORMAL)
            return
        
        label = classification['label']
        confidence = classification['confidence']
        
        self.update_output(f"【分类结果】{label} (置信度: {confidence:.1%})\n\n正在生成回复...")
        self.update_scores(scores_text)
        self.status_label.config(text="生成回复中...", foreground='orange')
        self.root.update()
        
        def get_reply_in_thread():
            try:
                reply = self.app.get_reply(user_input, classification)
                
                result = f"【分类结果】{label} (置信度: {confidence:.1%})\n\n【AI回复】\n{reply}"
                self.update_output(result)
                
                self.input_text.delete("1.0", tk.END)
                self.status_label.config(text="就绪", foreground='green')
            except Exception as e:
                self.update_output(f"错误: {str(e)}")
                self.status_label.config(text=f"错误", foreground='red')
                messagebox.showerror("错误", str(e))
            finally:
                self.send_btn.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=get_reply_in_thread)
        thread.daemon = True
        thread.start()
    
    def clear_chat(self):
        self.app.clear()
        self.update_output("对话历史已清空")
        self.update_scores("")
        self.status_label.config(text="已清空", foreground='green')


def main():
    root = tk.Tk()
    app = ChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
