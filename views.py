from flask import Flask, send_from_directory, request, jsonify, session
from flask_cors import CORS
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from flask_caching import Cache

my_app = Flask(__name__, static_folder='static')
my_app.secret_key = os.getenv("FLASK_SECRET_KEY")
my_app.config["CACHE_DEFAULT_TIMEOUT"] = 600
my_app.config["CACHE_TYPE"] = "simple"
cache = Cache(my_app)
CORS(my_app)

# # グローバル変数に性格診断の結果を一時的に保持
# latest_result = None
# scenario = None  # シナリオを保持
# user_job = None

@my_app.route('/')
def serve_frontend():
    return send_from_directory(my_app.static_folder, 'index.html')


# OpenAIの設定
# 環境変数
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
llm = ChatOpenAI(api_key=openai_api_key, base_url=openai_api_base, model="gpt-4o-mini", temperature=0)
file_path = os.path.join(os.path.dirname(__file__), 'learn_16personalities.txt')



#Embeddingを行うモデル
embeddings_model = OpenAIEmbeddings(
    openai_api_base=openai_api_base
)

#テキストファイルを読み込み
loader = TextLoader(file_path)
doc = loader.load()
#読み込んだ内容をチャンク化
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20) #分割するためのオブジェクト
splited = splitter.split_documents(doc)

#Embeddingを行うモデル
embeddings_model = OpenAIEmbeddings(
    openai_api_base=openai_api_base
)
#Emmbeddingの結果をCromaに保存、オブジェクトとして保存
vectorstore = Chroma.from_documents(documents=doc, embedding=embeddings_model)
#as_revectorstorメソッドでvectorstoreを検索機に変換、検索タイプはコサイン類似度、検索には1つのチャンクを参照、返すようにする
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})


#文章を1行の文字列にフォーマットする関数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#プロンプトを作成
template = """
以下の情報を参考に、ユーザの性格タイプについて以下の形式のように、箇条書きで簡単にまとめてください。
特徴、強み、弱みについてそれぞれ3つずつ挙げてください。
~ 特徴 ~
    -
    -
    -
~ 強み ~
    -
    -
    -
~ 弱み ~
    -
    -
    -
{context}

ユーザの性格タイプ: {latest_result}
"""
rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "latest_result": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

@my_app.route('/api/type', methods=["POST", "GET"])
@cache.cached(unless=lambda: request.method == 'POST')
def user_type():
    # global latest_result
    
    if request.method == "POST":
        rcv = request.get_json()
        result = ""
        
        # 各指標のスコアを計算
        score_e_i = sum(rcv.get("e_or_i", []))
        score_s_n = sum(rcv.get("s_or_n", []))
        score_t_f = sum(rcv.get("t_or_f", []))
        score_p_j = sum(rcv.get("p_or_j", []))
        
        # 性格タイプの判定
        if score_e_i >= 0:
            result += "E"
        else:
            result += "I"
            
        if score_s_n >= 0:
            result += "S"
        else:
            result += "N"
            
        if score_t_f >= 0:
            result += "T"
        else:
            result += "F"
            
        if score_p_j >= 0:
            result += "P"
        else:
            result += "J"

        # 診断結果を保持
        session['USERTYPE'] = result
            #print("診断結果が保存されました:", latest_result)
        return jsonify({"ready": True})

    elif request.method == "GET":
            #print("GETリクエストが呼び出されました")
        if 'USERTYPE' in session:
            ai_explains_type = rag_chain.invoke(session['USERTYPE'])
            return jsonify({"typeResult": session['USERTYPE'], "typeExplain": ai_explains_type})
        else:
            return jsonify({"error": "No result found"}), 404

#プロンプトを作成
template2 = """
    あなたは猫の占い師です。
    あなたにはユーザの性格を4文字のアルファベットで表す16タイプ性格診断の結果とユーザが将来なりたい職業が与えられます。
    ユーザの性格を踏まえて、ユーザが将来その職業についた際のシナリオを作成してください。
    シナリオは以下のコンテンツを含めてください。
    1. ユーザはその仕事に向いているか
    2. ユーザがその仕事に就いた際にうまくいくこと、苦労すること
    3. その仕事の現状とユーザがその仕事に就いた際の、起床から就寝までの1日のスケジュール
    4. 最後に、あなたが占い師としてユーザに伝えたいこと
    なお、シナリオは上記の1. ~ 4.について、1.、2.、3.、4.から始めてください。
    また、ユーザのことは"キミ"として、語尾は猫のように"ニャン"としてください。
    
    以下には、16性格タイプの詳細な情報が含まれていますので、参考にしてください。
    context: {context}
    
    {input}
"""
scenario_prompt = PromptTemplate.from_template(template2)


scenario_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | scenario_prompt
    | llm
    | StrOutputParser()
)    

@my_app.route('/api/scenario', methods=['POST', 'GET'])
@cache.cached(unless=lambda: request.method == 'POST')
def scenario_gen():
    # global scenario  # グローバル変数を参照
    # global user_job
    if request.method == 'POST':
        rcv = request.get_json()
        session['USERJOB'] = rcv.get("job", "")
        input = f"ユーザの性格タイプは{session['USERTYPE']}です。将来は{session['USERJOB']}になりたいと思っています。ユーザが将来{session['USERJOB']}に就いた時のシナリオを生成してください。"
        session["SCENARIO"] = scenario_chain.invoke(input)
        return jsonify({"scenarioReady": True})  # シナリオが生成されたことを示すフラグ
    elif request.method == 'GET':
        if 'SCENARIO' in session:
            return jsonify({"scenario": session["SCENARIO"], "type": session["USERTYPE"], "job": session["USERJOB"]})  # シナリオを返す
        else:
            return jsonify({"error": "No scenario found"}), 404  # シナリオがない場合
    
if __name__ == "__main__":
    my_app.run()