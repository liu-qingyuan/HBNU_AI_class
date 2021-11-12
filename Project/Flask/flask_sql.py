from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# 配置数据库地址
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:yydsjkhs@localhost/flask_sql_demo'

# 跟踪数据库修改(false)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 数据库对象
db = SQLAlchemy(app)

'''
两张表
角色（管理员 / 普通用户）
用户（角色ID）
'''

# 数据库的模型，需要继承db.Model
class Role(db.Model):
    # 定义表名
    __tablename__ = 'roles'
    
    # 定义字段
    # db.Column表示是一个字段
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(16), unique=True)

class User(db.Model):
    __tablename__ = 'users'
    id =db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(16))
    # db.ForeignKey('roles.id')  --> '外键'.'表名'
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))


    
@app.route('/')
def index():
    return 'Hello mysql'

if __name__ == '__main__':
    db.drop_all()

    db.create_all()


    app.run()
