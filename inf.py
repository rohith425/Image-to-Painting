from flask import Flask
from flask import Flask, render_template, flash, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed
from wtforms import SubmitField
import os
from cycle import run


app=Flask(__name__)
app.config['SECRET_KEY']='568325235'

picFolder= os.path.join('static','pics')
app.config['UPLOAD_FOLDER']=picFolder

class imgForm(FlaskForm):
    p_img = FileField(label='Input an Image',validators=[FileAllowed(['jpeg','png','jpg'])])
    submit = SubmitField(label='Submit')

def save_img(pic):
    picture_name='target.jpg'
    picture_path=os.path.join(app.root_path,'static/input',picture_name)
    pic.save(picture_path)
    return picture_name

@app.route('/',methods=['GET','POST'])
def home_page():
    form = imgForm()
    if form.validate_on_submit():
        #try:
            img_file_name=save_img(form.p_img.data)
            #calling the model here and saving its result in static/pics
            run()
            pics_temp=os.listdir(r'/Users/rohith/ChitraE/static/pics')
            pics_temp.sort()
            pics=[]
            inp=[]
            for i in (pics_temp):
                if(i=='target.jpg'):
                    inp.append(os.path.join(app.config['UPLOAD_FOLDER'],i))
                # print(os.listdir('C:/Users/Aneesh Kulkarni/web_dev/flask projects/web page for yolo/static/pics'))
                else:
                    pics.append(os.path.join(app.config['UPLOAD_FOLDER'],i))

            print(pics)
            return render_template('show.html',path=app.root_path,user_imgs=pics,length=len(pics),inp=inp,leng=len(inp))
        # except:
        #     print("wtf")
        #     flash('Please put in an image in valid format','error')
        #     return render_template('homepage.html',form=form)

    if(form.p_img.data is not None):
        flash('Please select an image only','error')  
    return render_template('homepage.html',form=form)

if __name__=="__main__":
    app.run()