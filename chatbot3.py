import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio
import json

class ExpertPanelChatbot:
    def __init__(self, openai_api_key: str, experts: dict):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        self.experts = experts
        self.expert_chains = self._initialize_expert_chains()
        self.summary_chain = self._initialize_summary_chain()
        
    def _initialize_expert_chains(self) -> dict:
        expert_chains = {}
        expert_prompt_template = """
        あなたは{expert_role}として、以下の質問/トピックについて専門的な意見を提供してください。
        
        質問/トピック: {question}
        
        回答の際は以下の点に注意してください：
        - あなたの専門分野の観点からの分析を提供すること
        - 具体的な根拠や例を示すこと
        - 潜在的な課題や機会を指摘すること
        
        専門家としての意見:
        """
        
        for expert_id, expert_desc in self.experts.items():
            prompt = PromptTemplate(
                template=expert_prompt_template,
                input_variables=["expert_role", "question"]
            )
            expert_chains[expert_id] = LLMChain(llm=self.llm, prompt=prompt)
        return expert_chains
    
    def _initialize_summary_chain(self) -> LLMChain:
        summary_prompt_template = """
        以下は異なる専門家から提供された意見です。これらの意見を統合し、バランスの取れた包括的な結論を導き出してください。
        
        元の質問/トピック: {original_question}
        
        専門家の意見:
        {expert_opinions}
        
        統合された結論:
        1. 主な合意点
        2. 重要な懸念事項
        3. 推奨されるアクション
        4. 追加で検討が必要な点
        
        結論:
        """
        
        summary_prompt = PromptTemplate(
            template=summary_prompt_template,
            input_variables=["original_question", "expert_opinions"]
        )
        return LLMChain(llm=self.llm, prompt=summary_prompt)
    
    async def get_expert_opinion(self, expert_id: str, expert_desc: str, question: str, progress_placeholder) -> str:
        """個々の専門家の意見を取得し、リアルタイムで表示する"""
        response = await self.expert_chains[expert_id].arun(
            expert_role=expert_desc,
            question=question
        )
        progress_placeholder.write(f"💡 **{expert_id.replace('_', ' ').title()}の意見:**")
        progress_placeholder.markdown(response)
        return response
    
    async def get_integrated_response(self, question: str, progress_placeholder) -> tuple:
        """専門家の意見を収集し、統合された回答を生成する"""
        expert_opinions = {}
        
        # 非同期で各専門家の意見を取得
        tasks = []
        for expert_id, expert_desc in self.experts.items():
            task = self.get_expert_opinion(expert_id, expert_desc, question, progress_placeholder)
            tasks.append(task)
        
        # すべての意見を収集
        opinions = await asyncio.gather(*tasks)
        expert_opinions = dict(zip(self.experts.keys(), opinions))
        
        # 意見を統合
        formatted_opinions = "\n\n".join([
            f"{expert_id.upper()}の意見:\n{opinion}"
            for expert_id, opinion in expert_opinions.items()
        ])
        
        progress_placeholder.write("📊 **統合された結論を生成中...**")
        final_response = await self.summary_chain.arun(
            original_question=question,
            expert_opinions=formatted_opinions
        )
        
        return final_response, expert_opinions

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'experts' not in st.session_state:
        st.session_state.experts = {
            "business_analyst": "ビジネスアナリスト。市場分析、収益性、ビジネス戦略の観点から分析を行う。",
            "tech_expert": "技術専門家。技術的な実現可能性、必要なリソース、開発期間について分析を行う。",
            "risk_manager": "リスクマネージャー。法的リスク、セキュリティリスク、運用リスクを評価する。",
            "customer_advocate": "カスタマーアドボケイト。顧客視点での価値、使いやすさ、需要を評価する。"
        }

def expert_manager():
    """専門家の設定を管理するUI"""
    st.subheader("専門家の設定")
    
    # 専門家の追加
    with st.expander("専門家を追加"):
        col1, col2 = st.columns(2)
        with col1:
            new_expert_id = st.text_input("専門家ID（英数字）", key="new_expert_id")
        with col2:
            new_expert_desc = st.text_area("専門家の説明", key="new_expert_desc")
        
        if st.button("追加", key="add_expert"):
            if new_expert_id and new_expert_desc:
                st.session_state.experts[new_expert_id] = new_expert_desc
                st.success(f"専門家「{new_expert_id}」が追加されました！")
    
    # 既存の専門家の管理
    with st.expander("専門家を管理"):
        for expert_id, expert_desc in list(st.session_state.experts.items()):
            st.markdown(f"**{expert_id}**")
            new_desc = st.text_area("説明", value=expert_desc, key=f"edit_{expert_id}")
            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("更新", key=f"update_{expert_id}"):
                    st.session_state.experts[expert_id] = new_desc
                    st.success("更新されました！")
            with cols[1]:
                if st.button("削除", key=f"delete_{expert_id}"):
                    del st.session_state.experts[expert_id]
                    st.warning("削除されました！")
    
    # 設定のインポート/エクスポート
    with st.expander("設定のインポート/エクスポート"):
        # エクスポート
        export_data = json.dumps(st.session_state.experts, indent=2, ensure_ascii=False)
        st.download_button(
            label="設定をエクスポート",
            data=export_data,
            file_name="expert_settings.json",
            mime="application/json"
        )
        
        # インポート
        uploaded_file = st.file_uploader("設定ファイルをインポート", type="json")
        if uploaded_file is not None:
            try:
                imported_experts = json.load(uploaded_file)
                st.session_state.experts = imported_experts
                st.success("設定がインポートされました！")
            except Exception as e:
                st.error(f"インポートエラー: {str(e)}")

async def main():
    st.title("カスタマイズ可能な専門家パネルチャットボット 🤖")
    
    init_session_state()
    
    # サイドバーで設定
    with st.sidebar:
        st.header("設定")
        api_key = st.text_input("OpenAI APIキー", type="password")
        if api_key:
            st.success("APIキーが設定されました！")
        
        # 専門家の管理UIを表示
        expert_manager()
    
    # メインのチャットインターフェース
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # チャット入力
    if prompt := st.chat_input("質問を入力してください"):
        if not api_key:
            st.error("OpenAI APIキーを設定してください！")
            return
        
        if not st.session_state.experts:
            st.error("少なくとも1人の専門家を設定してください！")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            progress_placeholder = st.empty()
            
            # 新しいチャットボットインスタンスを作成（最新の専門家設定を反映）
            chatbot = ExpertPanelChatbot(api_key, st.session_state.experts)
            
            # 専門家の意見を取得し、リアルタイムで表示
            response, expert_opinions = await chatbot.get_integrated_response(prompt, progress_placeholder)
            
            # 最終的な統合結論を表示
            st.write("🎯 **最終的な統合結論:**")
            st.write(response)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

if __name__ == "__main__":
    asyncio.run(main())