from flask import Flask,render_template, request, flash, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import datetime

# 目标：判断规定时间是不是签到
# 获取   签到时间 (当前时间)，判断   签到时间  是不是  早于  上课时间(规定时间)

app = Flask(__name__)

# 配置数据库地址
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:yydsjkhs@localhost/flask_class'

# 跟踪数据库修改(false)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 配置密钥
app.secret_key = 'qingyuan'

# 数据库对象
db = SQLAlchemy(app)

'''
1. 配置数据库
    a. 导入SQLAlchemy扩展
    b. 创建db对象, 配置参数
    c. 终端创建数据库 
2. 添加角色，学生，签到模型
'''

# 数据库的模型，需要继承db.Model
class Role(db.Model):
    # 定义表名
    __tablename__ = 'roles'
    
    # 定义字段
    # db.Column表示是一个字段
    role_id = db.Column(db.Integer, primary_key=True)
    role_class = db.Column(db.String(8), unique=True)
    
    def __repr__(self):
        return ('Role: %s' %(role_class))
    
class Student(db.Model):
    # 定义表名
    __tablename__ = 'students'
    
    # 定义字段
    # db.Column表示是一个字段
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(16), unique=False)
    # db.ForeignKey('roles.id')  --> '外键'.'表名'
    role_id = db.Column(db.Integer, db.ForeignKey('roles.role_id'))
    names = locals()
    for i in range(128):
        names['face_encoding%s' % i] = db.Column(db.DECIMAL(12, 10))
    
    # 关系引用
    roles = db.relationship('Role')
    
    def __repr__(self):
        return ('Student: %s %s' %(name, role_id))

class Sign(db.Model):
    __tablename__ = 'signs'
    sign_id =db.Column(db.Integer, primary_key=True)
    class_start_time = db.Column(db.DateTime)
    sign_in_time = db.Column(db.DateTime)
    is_sign_in = db.Column(db.Boolean)
    # db.ForeignKey('roles.id')  --> '外键'.'表名'
    sno = db.Column(db.Integer, db.ForeignKey('students.sno'))
    
    # 关系引用
    students = db.relationship('Student')

    def __repr__(self):
        return ('Book: %s %s %s %s' %(class_start_time, sign_in_time, is_sign_in, sno))


    
@app.route('/', methods=['POST','GET'])
def index():
    # 载入人脸数据库
    
    
    
    
    # 规定上课时间
    class_start_time = datetime.datetime.now().replace(hour=11, minute=30, second=0,microsecond=0)
    
    def is_sign(class_start_time, sign_in_time):
        '''
        格式: class_start_time, sign_in_time为datetime.datetime
        返回: True or False
        '''
        return class_start_time>sign_in_time
    
    
    # 生成签到数据
    def face_recognition_sign():
        # 获取对应人, 及识别时间
        
        # 如果数据库中有数据(2小时)      不写入
        # 如果数据库中没有数据(2小时)    写入
        
        pass
        
    # 查询所有学生信息
    stus = Student.query.all()
    
    
    now = datetime.datetime.now()
    sign1 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2019012604)
    now = datetime.datetime.now()
    sign2 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2017011837)
    now = datetime.datetime.now()
    sign3 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2019012621)
    now = datetime.datetime.now()
    sign4 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2019012606)
    now = datetime.datetime.now()
    sign5 = Sign(class_start_time=class_start_time+datetime.timedelta(hours=1), sign_in_time=now,is_sign_in =is_sign(class_start_time+datetime.timedelta(hours=1), now),sno=2019012604)
    # 把数据提交给用户会话
    db.session.add_all([sign1,sign2,sign3,sign4,sign5])
    # 提交会话
    db.session.commit()
    
    # 查询所有签到信息
    signs = Sign.query.all()
    
    # 传递信息给网页模板
    return render_template('index.html', signs=signs)


# # 删除书籍路由
# @app.route('/delete_book/<int:book_id>')
# def delete_book(book_id):
#     # 查询是否有此书
#     book = Book.query.get(book_id)

#     if book:
#         try:
#             db.session.delete(book)
#             db.session.commit()
#         except Exception as e:
#             print(e)
#             flash('删除书籍失败')
#             # 回滚
#             db.session.rollback()
#     else:
#         flash('没有此书籍')

#     # 重定向到主页
#     return redirect(url_for('index'))

# # 删除作者路由
# @app.route('/delete_author/<int:author_id>')
# def delete_author(author_id):
#     # 查询是否有此作者
#     author = Author.query.get(author_id)

#     if author:
#         # 先删除书再删除作者
#         try:
#             # 查询后直接删除
#             Book.query.filter_by(author_id=author.id).delete()
#             db.session.delete(author)
#             db.session.commit()
#         except Exception as e:
#             print(e)
#             flash('删除作者失败')
#             # 回滚
#             db.session.rollback()
#     else:
#         flash('没有此作者')

#     # 重定向到主页
#     return redirect(url_for('index'))

if __name__ == '__main__':
    db.drop_all()
    db.create_all()

    # 生成角色数据
    role1 = Role(role_class = '学生')
    role2 = Role(role_class = '老师')
    role3 = Role(role_class = '管理员')
    # 把数据提交给用户会话
    db.session.add_all([role1,role2,role3])
    # 提交会话
    db.session.commit()
    
    # 生成学生数据
    stu1 = Student(sno=2019012604,name = '小沅', role_id=role3.role_id, face_encoding0=-0.09634063)
    stu2 = Student(sno=2017011837,name = '小佳乐', role_id=role1.role_id)
    stu3 = Student(sno=2019012621,name = '小泰', role_id=role2.role_id)
    stu4 = Student(sno=2019012606,name = '小韧', role_id=role3.role_id)
    # 把数据提交给用户会话
    db.session.add_all([stu1,stu2,stu3,stu4])
    # 提交会话
    db.session.commit()
    
    
    # 规定上课时间
    class_start_time = datetime.datetime.now().replace(hour=12, minute=30, second=0,microsecond=0)
    
    def is_sign(class_start_time, sign_in_time):
        '''
        格式: class_start_time, sign_in_time为datetime.datetime
        返回: True or False
        '''
        return class_start_time>sign_in_time
    
    # 生成签到数据
    now = datetime.datetime.now()
    sign1 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=stu1.sno)
    now = datetime.datetime.now()
    sign2 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2017011837)
    now = datetime.datetime.now()
    sign3 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2019012621)
    now = datetime.datetime.now()
    sign4 = Sign(class_start_time=class_start_time, sign_in_time=now,is_sign_in =is_sign(class_start_time, now),sno=2019012606)
    now = datetime.datetime.now()
    sign5 = Sign(class_start_time=class_start_time+datetime.timedelta(hours=2), sign_in_time=now,is_sign_in =is_sign(class_start_time+datetime.timedelta(hours=2), now),sno=2019012604)
    # 把数据提交给用户会话
    db.session.add_all([sign1,sign2,sign3,sign4,sign5])
    # 提交会话
    db.session.commit()
    
    app.run()
