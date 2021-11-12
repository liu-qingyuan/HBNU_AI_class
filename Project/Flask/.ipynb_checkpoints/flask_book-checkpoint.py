from flask import Flask,render_template, request, flash, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)

# 配置数据库地址
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:yydsjkhs@localhost/flask_books'

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
2. 添加书和作者模型
'''

# 数据库的模型，需要继承db.Model
class Author(db.Model):
    # 定义表名
    __tablename__ = 'authors'
    
    # 定义字段
    # db.Column表示是一个字段
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(16), unique=True)
    
    # 关系引用
    books = db.relationship('Book', backref='author')
    
    def __repr__(self):
        return ('Author: %s' %(name))

class Book(db.Model):
    __tablename__ = 'books'
    id =db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(16))
    # db.ForeignKey('roles.id')  --> '外键'.'表名'
    author_id = db.Column(db.Integer, db.ForeignKey('authors.id'))

    def __repr__(self):
        return ('Book: %s %s' %(name, author_id))

# 自定义表单类
class AuthorForm(FlaskForm):
    author = StringField('作者', validators=[DataRequired()])
    book = StringField('书籍', validators=[DataRequired()])
    submit = SubmitField('提交')
    
    
    
    
@app.route('/', methods=['POST','GET'])
def index():
    # 创建自定义的表单类
    author_form = AuthorForm()
    
    '''
    逻辑判断:
    '''
    if author_form.validate_on_submit():
        
        # 验证通过获取数据
        author_name = author_form.author.data
        book_name = author_form.book.data
        
        # 判断作者是否存在
        author = Author.query.filter_by(name=author_name).first()
        
        # 如果作者存在
        if author:
            # 判断书籍是不是存在
            book = Book.query.filter_by(name=book_name).first()
            # 如果存在
            if book:
                flash('已存在同名书籍')
            # 如果不存在，try,except
            else:
                try:
                    book = Book(name=book_name, author_id=author.id)
                    # 把数据提交给用户会话
                    db.session.add(book)
                    # 提交会话
                    db.session.commit()
                except Exception as e:
                    print(e)
                    flash('添加书籍失败')
                    # 回滚
                    db.session.rollback()
        else:
            # 如果作者不存在，添加作者和书籍
            try:
                # 先把作者上传到数据库
                author = Author(name=author_name)
                # 把数据提交给用户会话
                db.session.add(author)
                # 提交会话
                db.session.commit()
                
                # 再把书籍上传到数据库
                book = Book(name=book_name, author_id=author.id)
                # 把数据提交给用户会话
                db.session.add(book)
                # 提交会话
                db.session.commit()
            except Exception as e:
                print(e)
                flash('添加作者失败')
                # 回滚
                db.session.rollback()
    else:
        if request.method == 'POST':
            flash('参数不全')
            
    # 查询所有作者信息
    authors = Author.query.all()
    
    # 传递信息给网页模板
    return render_template('index.html', authors=authors, form=author_form)


# 删除书籍路由
@app.route('/delete_book/<int:book_id>')
def delete_book(book_id):
    # 查询是否有此书
    book = Book.query.get(book_id)

    if book:
        try:
            db.session.delete(book)
            db.session.commit()
        except Exception as e:
            print(e)
            flash('删除书籍失败')
            # 回滚
            db.session.rollback()
    else:
        flash('没有此书籍')

    # 重定向到主页
    return redirect(url_for('index'))

# 删除作者路由
@app.route('/delete_author/<int:author_id>')
def delete_author(author_id):
    # 查询是否有此作者
    author = Author.query.get(author_id)

    if author:
        # 先删除书再删除作者
        try:
            # 查询后直接删除
            Book.query.filter_by(author_id=author.id).delete()
            db.session.delete(author)
            db.session.commit()
        except Exception as e:
            print(e)
            flash('删除作者失败')
            # 回滚
            db.session.rollback()
    else:
        flash('没有此作者')

    # 重定向到主页
    return redirect(url_for('index'))

if __name__ == '__main__':
    db.drop_all()
    db.create_all()

    # 生成作者数据
    au1 = Author(name = '大哥')
    au2 = Author(name = 'lbw')
    au3 = Author(name = '隔壁老王')
    # 把数据提交给用户会话
    db.session.add_all([au1, au2, au3])
    # 提交会话
    db.session.commit()
    
    # 生成图书数据
    bk1 = Book(name = '异世界的兔子女王', author_id=au1.id)
    bk2 = Book(name = '重生之我在LNG打上单', author_id=au2.id)
    bk3 = Book(name = '爱帮忙的好邻居日记', author_id=au3.id)
    bk4 = Book(name = '怎样征服美少女战士', author_id=au1.id)
    bk5 = Book(name = '如何让自己更猛', author_id=au1.id)
    bk6 = Book(name = '论水管修理与衣柜透风技术', author_id=au3.id)
    # 把数据提交给用户会话
    db.session.add_all([bk1, bk2, bk3, bk4, bk5, bk6])
    # 提交会话
    db.session.commit()
    
    app.run()