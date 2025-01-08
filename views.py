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
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

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

# データベースの実装
# Postgresql 環境変数からデータベースのURIを設定
my_app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')

my_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(my_app)

# Accountテーブルの定義
class Account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    password = db.Column(db.Text, nullable=False)
    
    def __repr__(self):
        return f'<Account {self.username}>'

# データベースを初期化
def initialize_database():
    with my_app.app_context():
        db.create_all()

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

        return jsonify({"ready": True})

    elif request.method == "GET":
        if result != "":
            ai_explains_type = rag_chain.invoke(result)
            return jsonify({"typeResult": result, "typeExplain": ai_explains_type})
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

@my_app.route('/api/register', methods=["POST"])
def account_register():
    try:
        # JSONデータを取得
        data = request.get_json()
        form_name = data.get("name")
        form_password = data.get("password")
        
        # 入力データの検証
        if not form_name or not form_password:
            return jsonify({"error": "Missing name or password"}), 400
        
        # ユーザーネームが既存か確認
        existing_account = Account.query.filter_by(username=form_name).first()
        if existing_account:
            return jsonify({"nameExist": False}), 409  # 重複エラー
        
        # アカウントの作成
        hashed_password = generate_password_hash(form_password)
        new_account = Account(username=form_name, password=hashed_password)
        db.session.add(new_account)
        db.session.commit()
        return jsonify({"nameExist": True}), 201  # 正常終了

    except Exception as e:
        # エラーログの出力
        my_app.logger.error(f"Error during account registration: {e}")
        db.session.rollback()
        return jsonify({"error": "An internal error occurred"}), 500

@my_app.route('/api/login', methods=["POST"])
def account_login():
    try:
        # JSONデータを取得
        data = request.get_json()
        account_name = data.get("name")
        account_password = data.get("password")
        
        # 入力データの検証
        if not account_name or not account_password:
            return jsonify({"error": "Missing name or password"}), 400
        
        # ユーザーネームが存在するか確認
        existing_account = Account.query.filter_by(username=account_name).first()
        if not existing_account:
            return jsonify({"loginSuccess": False, "error": "Account does not exist"}), 404  # アカウントが存在しない
        
        # パスワードが正しいか確認
        if not check_password_hash(existing_account.password, account_password):
            return jsonify({"loginSuccess": False, "error": "Incorrect password"}), 401  # パスワードが違う
        
        # ログイン成功
        return jsonify({"loginSuccess": True, "name": account_name}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # サーバーエラー



if __name__ == '__main__':
    initialize_database()
    my_app.run()