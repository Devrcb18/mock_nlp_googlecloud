from flask import Flask, request, jsonify, render_template
import joblib
import lightgbm as lgb

app = Flask(__name__)
model = lgb.Booster(model_file=r"C:\Users\devan\nlp\project1\model (1).txt")
tf = joblib.load(r"C:\Users\devan\nlp\project1\vectorizer (1).joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form.get('text_input', '')
    
    if not sentence:
        return render_template('index.html', prediction_text='Please enter some text')
    
    corpus = [sentence]
    s = tf.transform(corpus)
    x = s.toarray()
    p = model.predict(x)
    v = p[0]
    if v > 0.5:
        op = 'OFFENSIVE CONTENT DETECTED'
        result_class = 'offensive'
    else:
        op = 'CONTENT IS CLEAN'
        result_class = 'clean'
    
    return render_template('index.html',prediction_text=op,result_class=result_class,show_result=True)

if __name__ == '__main__':
    app.run(debug=True)