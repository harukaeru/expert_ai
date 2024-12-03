import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio
import json

class ExpertPanelChatbot:
    def __init__(self, openai_api_key: str, experts: dict, model_config: dict):
        self.llm = ChatOpenAI(
            model_name=model_config.get("model_name", "gpt-4o-mini"),
            temperature=model_config.get("temperature", 0.7),
            openai_api_key=openai_api_key
        )
        
        self.experts = {k: v["description"] for k, v in experts.items()}
        self.expert_chains = self._initialize_expert_chains()
        self.summary_chain = self._initialize_summary_chain()

    def _initialize_expert_chains(self) -> dict:
        expert_chains = {}
        expert_prompt_template = """
        あなたは{expert_role}として、以下の質問/トピックについて意見を提供してください。
        ユーザーの質問に対してあなたの立場・あなたの観点から徹底的に考察してください。
        わからない部分に関しては、素直に「わからない」と答えてください。
        
        質問/トピック: {question}
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
        progress_placeholder.markdown(f"🤔 **{expert_id.replace('_', ' ').title()}** が考えています...")
        
        response = await self.expert_chains[expert_id].arun(
            expert_role=expert_desc,
            question=question
        )
        
        # 専門家の回答を整形して表示
        formatted_response = f"""
        ### 💡 {expert_id.replace('_', ' ').title()}の意見:
        
        {response}
        
        ---
        """

        progress_placeholder.markdown(formatted_response)
        return response
    
    async def get_integrated_response(self, question: str, expert_opinions: dict, progress_placeholder) -> tuple:
        """専門家の意見を収集し、統合された回答を生成する"""
        # 意見を統合
        formatted_opinions = "\n\n".join([
             f"{expert_id.upper()}の意見:\n{opinion}"
             for expert_id, opinion in expert_opinions.items()
        ])
    
        progress_placeholder.markdown("### 🔄 最終的な結論を生成中...")
        final_response = await self.summary_chain.arun(
            original_question=question,
            expert_opinions=formatted_opinions
        )
    
        return final_response

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'experts' not in st.session_state:
        st.session_state.experts = {
            "graph_specialist": {
                "description": "グラフ構造の専門家。アルゴリズムやネットワークに詳しくいつもその観点でものごとを考えて分析を行う。",
                "avatar": "🕸️",
                "name": "グラフ理論専門家"
            },
            "tech_expert": {
                "description": "技術専門家。技術的な実現可能性、必要なリソース、開発期間について分析を行う。",
                "avatar": "👨‍💻",
                "name": "技術アーキテクト"
            },
            "math_expert": {
                "description": "数学者。幅広い分野において習熟しており、とりわけ応用数学に強みがある。定式化が好き。数学の観点からものごとを分析する。",
                "avatar": "📐",
                "name": "応用数学者"
            },
            "money_hunter": {
                "description": "富豪かつ経済学者。常にそれが資産になるか、あるいはそれが資産になるために何をすべきかを考えて分析する。金融や経済が大好き。",
                "avatar": "💰",
                "name": "投資家兼経済学者"
            },
            "layman_takehashi": {
                "description": "ユーザーと仲のいい友達である竹橋。頭はいいがほとんどの場面において素人で、社会のことがよくわかっていないが、それでもユーザーの質問に対して一生懸命自分の立場から徹底的に考察してくれる。",
                "avatar": "🙆",
                "name": "マイフレンド竹橋"
            },

        }
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7
        }

def model_config_manager():
    """モデル設定を管理するUI"""
    st.subheader("モデル設定")
    
    # 利用可能なモデルのリスト
    available_models = [
        "gpt-4o-mini",  # GPT-4o-mini
        "gpt-4o",               # GPT-4o
    ]
    
    # モデル選択
    selected_model = st.selectbox(
        "モデルを選択",
        available_models,
        index=available_models.index(st.session_state.model_config["model_name"])
    )
    
    # Temperature設定
    temperature = st.slider(
        "Temperature (創造性の度合い)",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.model_config["temperature"],
        step=0.1,
        help="低い値: より一貫性のある応答\n高い値: よりクリエイティブな応答"
    )
    
    # 設定の保存
    if selected_model != st.session_state.model_config["model_name"] or \
       temperature != st.session_state.model_config["temperature"]:
        st.session_state.model_config["model_name"] = selected_model
        st.session_state.model_config["temperature"] = temperature
        st.success("モデル設定が更新されました！")
    
    # モデル情報の表示
    with st.expander("モデル情報"):
        st.markdown("""
        **モデルの特徴:**
        - **GPT-4o-mini**: 高性能な基本モデル。複雑なタスクに適しています。コストは小程度です。
        - **GPT-4o**: 最新で最も高性能なモデル。コストは高めですが、最新の知識と高い処理能力を持ちます。
        
        **Temperature設定の影響:**
        - **0.0-0.3**: より事実に基づいた、決定論的な応答
        - **0.4-0.7**: バランスの取れた応答
        - **0.8-2.0**: よりクリエイティブで多様な応答
        """)
    
    # 設定のインポート/エクスポート
    with st.expander("モデル設定のインポート/エクスポート"):
        # エクスポート
        export_data = json.dumps(st.session_state.model_config, indent=2, ensure_ascii=False)
        st.download_button(
            label="モデル設定をエクスポート",
            data=export_data,
            file_name="model_settings.json",
            mime="application/json"
        )
        
        # インポート
        uploaded_file = st.file_uploader("モデル設定ファイルをインポート", type="json")
        if uploaded_file is not None:
            try:
                imported_config = json.load(uploaded_file)
                if "model_name" in imported_config and "temperature" in imported_config:
                    st.session_state.model_config = imported_config
                    st.success("モデル設定がインポートされました！")
                else:
                    st.error("無効な設定ファイルです")
            except Exception as e:
                st.error(f"インポートエラー: {str(e)}")


def expert_manager():
    """専門家の設定を管理するUI"""
    st.subheader("専門家の設定")
  
    # 専門家の追加
    with st.expander("専門家を追加"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_expert_id = st.text_input("専門家ID（英数字）", key="new_expert_id")
        with col2:
            new_expert_name = st.text_input("表示名", key="new_expert_name")
        with col3:
            new_expert_avatar = st.text_input("アバター絵文字", key="new_expert_avatar")
        new_expert_desc = st.text_area("専門家の説明", key="new_expert_desc")
        
        if st.button("追加", key="add_expert"):
            if all([new_expert_id, new_expert_name, new_expert_avatar, new_expert_desc]):
                st.session_state.experts[new_expert_id] = {
                    "description": new_expert_desc,
                    "avatar": new_expert_avatar,
                    "name": new_expert_name
                }
                st.success(f"専門家「{new_expert_name}」が追加されました！")
  
    # 既存の専門家の管理
    with st.expander("専門家を管理"):
        for expert_id, expert_info in list(st.session_state.experts.items()):
            st.markdown(f"**{expert_info['name']} {expert_info['avatar']}**")
            cols = st.columns(3)
            with cols[0]:
                new_name = st.text_input("表示名", value=expert_info['name'], key=f"name_{expert_id}")
            with cols[1]:
                new_avatar = st.text_input("アバター", value=expert_info['avatar'], key=f"avatar_{expert_id}")
            new_desc = st.text_area("説明", value=expert_info['description'], key=f"desc_{expert_id}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("更新", key=f"update_{expert_id}"):
                    st.session_state.experts[expert_id] = {
                        "description": new_desc,
                        "avatar": new_avatar,
                        "name": new_name
                    }
                    st.success("更新されました！")
            with col2:
                if st.button("削除", key=f"delete_{expert_id}"):
                    del st.session_state.experts[expert_id]
                    st.warning("削除されました！")
            st.markdown("---")

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
    st.title("専門家パネルチャットボット 🤖")
    
    init_session_state()
    
    # サイドバーで設定
    with st.sidebar:
        st.header("設定")
        
        # OpenAI APIキー設定
        api_key = st.text_input("OpenAI APIキー", type="password")
        if api_key:
            st.success("APIキーが設定されました！")
        
        # タブで設定を整理
        tab1, tab2 = st.tabs(["モデル設定", "専門家設定"])
        
        with tab1:
            model_config_manager()
        
        with tab2:
            expert_manager()  # 前回のコードから変更なし
    
    # メインのチャットインターフェース
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        elif message["role"] == "expert":
            # 専門家のメッセージ
            expert_info = st.session_state.experts[message["expert_id"]]
            with st.chat_message("assistant", avatar=expert_info["avatar"]):
                st.markdown(f"**{expert_info['name']}** の分析:")
                st.markdown(message["content"])
        elif message["role"] == "summary":
            # 最終統合意見
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("**統合された結論:**")
                st.markdown(message["content"])
                st.markdown(f"*使用モデル: {st.session_state.model_config['model_name']} (Temperature: {st.session_state.model_config['temperature']})*")
    
    # チャット入力と処理
    if prompt := st.chat_input("質問を入力してください"):
        if not api_key:
            st.error("OpenAI APIキーを設定してください！")
            return
        
        if not st.session_state.experts:
            st.error("少なくとも1人の専門家を設定してください！")
            return
        
        # ユーザーの質問を表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # チャットボットインスタンスを作成
        chatbot = ExpertPanelChatbot(
            api_key,
            st.session_state.experts,
            st.session_state.model_config
        )
        
        # 各専門家の意見を順番に取得して表示
        expert_opinions = {}
        for expert_id, expert_info in st.session_state.experts.items():
            # 専門家が考えていることを表示
            thinking_message = f"💭 **{expert_info['name']}** が分析中..."
            with st.chat_message("assistant", avatar=expert_info["avatar"]):
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown(thinking_message)
                
                # 意見を取得
                opinion = await chatbot.get_expert_opinion(
                    expert_id,
                    expert_info["description"],
                    prompt,
                    thinking_placeholder,
                )
                expert_opinions[expert_id] = opinion
                
                # 考え中のメッセージを実際の意見で置き換え
                thinking_placeholder.empty()
                st.markdown(f"**{expert_info['name']}** の分析:")
                st.markdown(opinion)
            
            # メッセージを履歴に追加
            st.session_state.messages.append({
                "role": "expert",
                "expert_id": expert_id,
                "content": opinion
            })
        
        # 統合された結論を生成
        with st.chat_message("assistant", avatar="🤖"):
            final_response = await chatbot.get_integrated_response(prompt, expert_opinions, st)
        
              # 統合された結論を表示
            summary_message = {
                "role": "summary",
                "content": final_response
            }
            st.session_state.messages.append(summary_message)

            st.markdown("**統合された結論:**")
            st.markdown(final_response)
            st.markdown(f"*使用モデル: {st.session_state.model_config['model_name']} (Temperature: {st.session_state.model_config['temperature']})*")

if __name__ == "__main__":
    asyncio.run(main())