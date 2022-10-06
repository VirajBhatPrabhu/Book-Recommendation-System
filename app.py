from flask import Flask,render_template,request
import pickle
import numpy as np



topbooks = pickle.load(open('topbooks.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
pivot_df = pickle.load(open('pivot_df.pkl','rb'))
sim_scores = pickle.load(open('sim_scores.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(topbooks['Book-Title'].values),
                           author = list(topbooks['Book-Author'].values),
                           image = list(topbooks['Image-URL'].values),
                           votes = list(topbooks['Num_ratings'].values),
                           rating = list(topbooks['Avg_ratings'].values))
@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pivot_df.index == user_input)[0][0]
    similarity = sorted(list(enumerate(sim_scores[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similarity:
        item = []
        temp_df = books[books['Book-Title'] == pivot_df.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL'].values))

        data.append(item)
    print(data)

    return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)