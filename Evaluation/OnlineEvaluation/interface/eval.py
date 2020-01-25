import json
import time
import zlib

from flask import Flask, render_template, request, redirect
app = Flask(__name__)

with open('/home/ashwath/Programs/OnlineEvaluation/online_eval_recommendations_100.json') as f:
    items = json.load(f)

def get_uid(request):
    ip = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    lang = request.headers.get('Accept-Language')
    uid = '{:x}'.format(
        zlib.adler32(str.encode('{}{}{}'.format(ip, user_agent, lang)))
        )
    return uid

def log_eval(uid, n, eval_json):
    timestamp = str(int(time.time()))
    with open('eval_log_{}'.format(uid), 'a') as f:
        line = '{}\n'.format('\u241f'.join([timestamp, uid, str(n), eval_json]))
        f.write(line)

@app.route('/')
def index():
    return redirect('/0/')

@app.route('/<item_num>/', methods=['GET'])
def eval(item_num):
    num_items = len(items)
    n = int(item_num)
    item = items[n]
    context = item['context'].replace('MAINCIT', '<span style="color: #d11; font-weight: bold;">MAINCIT</span>')
    recs_hd2v = item['hd2v']
    recs_bm25 = item['bm25']
    recs_hybrid = item.get('hybrid')
    return render_template('index.html', context=context, recs_hd2v=recs_hd2v, recs_bm25=recs_bm25, recs_hybrid=recs_hybrid, n=n, num_items=num_items)

@app.route('/<item_num>/pass', methods=['GET'])
def skip(item_num):
    uid = get_uid(request)
    n = int(item_num)
    num_items = len(items)
    log_eval(uid, n, json.dumps('pass'))
    return redirect('/{}/'.format(min(n+1, num_items-1)))

@app.route('/<item_num>/undecidable', methods=['GET'])
def undecidable(item_num):
    uid = get_uid(request)
    n = int(item_num)
    num_items = len(items)
    log_eval(uid, n, json.dumps('undecidable'))
    return redirect('/{}/'.format(min(n+1, num_items-1)))

@app.route('/<item_num>/rate', methods=['POST'])
def rate(item_num):
    uid = get_uid(request)
    n = int(item_num)
    num_items = len(items)
    log_eval(uid, n, json.dumps(request.form))
    return redirect('/{}/'.format(min(n+1, num_items-1)))

@app.route('/<item_num>/showmyratings', methods=['GET'])
def showmyratings(item_num):
    uid = get_uid(request)
    try:
        with open('eval_log_{}'.format(uid)) as f:
            csv = f.read()
            head = '{}\n'.format('\u241f'.join(['timestamp', 'user id', 'item number', 'rating']))
            csv = head + csv
    except FileNotFoundError:
        csv = 'no ratings recorded'
    return csv.replace('\n', '<br>')
