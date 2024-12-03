import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio
import os

class ExpertPanelChatbot:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        self.experts = {
            "business_analyst": "ビジネスアナリスト。市場分析、収益性、ビジネス戦略の観点から分析を行う。",
            "tech_expert": "技術専門家。技術的な実現可能性、必要なリソース、開発期間について分析を行う。",
            "risk_manager": "リスクマネージャー。法的リスク、セキュリティリスク、運用リスクを評価する。",
            "customer_advocate": "カスタマーアドボケイト。顧客視点での価値、使いやすさ、需要を評価する。"
        }
        
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
    
    async def get_expert_opinions(self, question: str) -> dict:
        opinions = {}
        for expert_id, expert_desc in self.experts.items():
            response = await self.expert_chains[expert_id].arun(
                expert_role=expert_desc,
                question=question
            )
            opinions[expert_id] = response
        return opinions
    
    async def get_integrated_response(self, question: str) -> str:
        expert_opinions = await self.get_expert_opinions(question)
        formatted_opinions = "\n\n".join([
            f"{expert_id.upper()}の意見:\n{opinion}"
            for expert_id, opinion in expert_opinions.items()
        ])
        
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

async def main():
    st.title("専門家パネルチャットボット 🤖")
    
    init_session_state()
    
    # サイドバーでAPIキーを設定
    with st.sidebar:
        st.header("設定")
        api_key = st.text_input("OpenAI APIキー", type="password")
        if api_key:
            st.session_state.chatbot = ExpertPanelChatbot(api_key)
            st.success("APIキーが設定されました！")
    
    # メインのチャットインターフェース
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "expert_opinions" in message:
                    with st.expander("各専門家の詳細な意見を見る"):
                        for expert, opinion in message["expert_opinions"].items():
                            st.subheader(f"💡 {expert.replace('_', ' ').title()}")
                            st.write(opinion)
    
    # チャット入力
    if prompt := st.chat_input("質問を入力してください"):
        if not api_key:
            st.error("OpenAI APIキーを設定してください！")
            return
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("専門家に意見を聞いています..."):
                response, expert_opinions = await st.session_state.chatbot.get_integrated_response(prompt)
                st.write(response)
                with st.expander("各専門家の詳細な意見を見る"):
                    for expert, opinion in expert_opinions.items():
                        st.subheader(f"💡 {expert.replace('_', ' ').title()}")
                        st.write(opinion)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "expert_opinions": expert_opinions
                })

if __name__ == "__main__":
    asyncio.run(main())