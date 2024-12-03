from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document
from typing import List
import os

class ExpertPanelChatbot:
    def __init__(self, openai_api_key: str):
        # 基本のLLMモデルを設定
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # 各専門家の役割を定義
        self.experts = {
            "business_analyst": "ビジネスアナリスト。市場分析、収益性、ビジネス戦略の観点から分析を行う。",
            "tech_expert": "技術専門家。技術的な実現可能性、必要なリソース、開発期間について分析を行う。",
            "risk_manager": "リスクマネージャー。法的リスク、セキュリティリスク、運用リスクを評価する。",
            "customer_advocate": "カスタマーアドボケイト。顧客視点での価値、使いやすさ、需要を評価する。"
        }
        
        # 各専門家のチェーンを初期化
        self.expert_chains = self._initialize_expert_chains()
        
        # 意見統合用のチェーンを初期化
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
            
            expert_chains[expert_id] = LLMChain(
                llm=self.llm,
                prompt=prompt
            )
            
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
        
        return LLMChain(
            llm=self.llm,
            prompt=summary_prompt
        )
    
    async def get_expert_opinions(self, question: str) -> dict:
        """各専門家から意見を収集する"""
        opinions = {}
        
        for expert_id, expert_desc in self.experts.items():
            response = await self.expert_chains[expert_id].arun(
                expert_role=expert_desc,
                question=question
            )
            opinions[expert_id] = response

        return opinions
    
    async def get_integrated_response(self, question: str) -> str:
        """専門家の意見を収集し、統合された回答を生成する"""
        # 専門家の意見を収集
        expert_opinions = await self.get_expert_opinions(question)
        
        # 意見を統合しやすい形式に整形
        formatted_opinions = "\n\n".join([
            f"{expert_id.upper()}の意見:\n{opinion}"
            for expert_id, opinion in expert_opinions.items()
        ])

        print(formatted_opinions)
        
        # 統合された回答を生成
        final_response = await self.summary_chain.arun(
            original_question=question,
            expert_opinions=formatted_opinions
        )
        
        return final_response

# 使用例
async def main():
    # OpenAI APIキーを設定
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # チャットボットを初期化
    chatbot = ExpertPanelChatbot(openai_api_key)
    
    # 質問例
    question = "新しいeコマースプラットフォームを立ち上げることを検討しています。どのような点に注意すべきでしょうか？"
    
    # 統合された回答を取得
    response = await chatbot.get_integrated_response(question)
    print(response)

# 非同期実行
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())