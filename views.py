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
    
# Uranaiテーブルの定義
class Uranai(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.Integer, db.ForeignKey('account.id'), nullable=False)  # 外部キー
    user_type = db.Column(db.String(4), nullable=False)  # 性格タイプ
    user_job = db.Column(db.String(25), nullable=False)  # 職業
    scenario = db.Column(db.Text, nullable=False)  # シナリオ内容

    # Accountテーブルとのリレーション
    account = db.relationship('Account', backref=db.backref('uranai_entries', lazy=True))

    def __repr__(self):
        return f'<Uranai for Account {self.account_id}>'
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

@my_app.route('/api/type', methods=["POST"])
def user_type():
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
        
    ai_explains_type = rag_chain.invoke(result)
    return jsonify({"ready": True, "result": result, "typeExplain": ai_explains_type})

@my_app.route("/api/knowntype", methods=["POST"])
def user_knows_type():
    rcv = request.get_json()
    result = rcv.get("type", "")
    ai_explains_type = rag_chain.invoke(result)
    return jsonify({"ready": True, "result": result, "typeExplain": ai_explains_type})


@my_app.route('/api/judge', methods=["POST"])
def user_type_explain():
    rcv = request.get_json() # resultをフロントエンドから受け取る
    result = rcv.get("result", [])
    if result != "":
        ai_explains_type = rag_chain.invoke(result) # 処理
        return jsonify({"typeResult": result, "typeExplain": ai_explains_type}) # 結果を返す
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

@my_app.route('/api/scenario', methods=['POST'])
def scenario_gen():
    rcv = request.get_json()
    user_type = rcv.get("type", "")
    user_job = rcv.get("job", "")
    user_name = rcv.get("name", "")

    # ログインしていない場合
    if user_name == "":
        scenario = scenario_chain.invoke(f"ユーザの性格タイプは{user_type}です。将来は{user_job}になりたいと思っています。ユーザが将来{user_job}に就いた時のシナリオを生成してください。")
        return jsonify({"scenarioReady": True, "scenario": scenario})  # シナリオのみを返す

    # ログインしている場合
    account = Account.query.filter_by(username=user_name).first()
    if not account:
        return jsonify({"error": "Account not found"}), 404

    # シナリオ生成
    scenario = scenario_chain.invoke(f"ユーザの性格タイプは{user_type}です。将来は{user_job}になりたいと思っています。ユーザが将来{user_job}に就いた時のシナリオを生成してください。")

    # データベース保存
    generated = Uranai(account_id=account.id, user_job=user_job, user_type=user_type, scenario=scenario)
    db.session.add(generated)
    db.session.commit()

    return jsonify({"scenarioReady": True, "scenario": scenario})  # シナリオが生成されたことを示すフラグ

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

@my_app.route('/api/past', methods=["POST"])
def past_uranai():
    try:
        # JSONデータを取得
        info = request.get_json()
        account_name = info.get("name")
        
        if not account_name:
            return jsonify({"error": "Account name is required"}), 400

        # Accountテーブルからusernameで該当アカウントを取得
        account = Account.query.filter_by(username=account_name).first()
        if not account:
            return jsonify({"error": "User not found"}), 404

        # Uranaiテーブルからすべてのデータを取得
        past_entries = Uranai.query.filter_by(account_id=account.id).all()

        # 結果をリスト形式でまとめる
        past = {
            "uranai_user_type": [entry.user_type for entry in past_entries],
            "uranai_user_job": [entry.user_job for entry in past_entries],
            "uranai_user_scenario": [entry.scenario for entry in past_entries]
        }

        return jsonify({"past": past})

    except Exception as e:
        my_app.logger.error(f"Error in past_uranai: {e}")
        return jsonify({"error": "An internal error occurred"}), 500
    


if __name__ == '__main__':
    initialize_database()
    my_app.run()