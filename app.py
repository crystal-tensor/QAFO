from flask import Flask, render_template, request, send_from_directory
from qtnm_model import QTNM_QAFO, load_language
from datetime import datetime
import os

app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = 'output'

# 语言设置
@app.context_processor
def inject_language():
    lang = request.args.get('lang', 'zh')
    return dict(lang=lang, trans=load_language(lang))

@app.route('/')
def index():
    lang = request.args.get('lang', 'zh')
    model = QTNM_QAFO(lang)
    results = model.generate_all_outputs()
    
    # 获取菜单详情
    menus = []
    menu_labels = [model.trans['conservative'], model.trans['balanced'], model.trans['ambitious']]
    for i, menu in enumerate(results['menus']):
        d = menu['decision']
        menus.append({
            'label': menu_labels[i],
            'us_score': menu['us'],
            'cn_score': menu['cn'],
            'delta_t': d['delta_t'],
            'tau_sum': d['tau'].sum(),
            'ntb_sum': d['ntb'].sum(),
            'tech_sum': d['tech'].sum(),
            'access_sum': d['access'].sum()
        })
    
    # 获取Q-NFI数据
    qnfi_data = results['qnfi'].to_dict('records')
    
    return render_template('index.html', 
                           menus=menus,
                           qnfi_data=qnfi_data,
                           gif_path='images/qafo_optimization.gif',
                           now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/static/<path:path>')
def static_file(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)