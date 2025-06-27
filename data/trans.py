import csv
import json

# 输入 JSON 数据（可替换为文件读取）
with open(r'D:\vscodework\lottery_predictor\data\raw\lottery_data.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# 解析 JSON
# data = json.loads(json_data)
data =json_data

# 写入 CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # 写入表头
    writer.writerow(['issue', 'date', 'numbers', 'sales', 'pool'])
    
    # 处理每条记录
    for item in data:
        # 格式化红球（补零）
        red_balls = [str(num).zfill(2) for num in item['red_balls']]
        
        # 格式化蓝球（补零）
        blue_ball = str(item['blue_ball']).zfill(2)
        
        # 组合号码字段
        numbers = ','.join(red_balls) + '+' + blue_ball
        
        # 写入 CSV 行
        writer.writerow([
            item['period'],
            item['date'],
            numbers,
            item['sales_amount'],
            item['pool_amount']
        ])

print("CSV 文件已生成成功！")