from bs4 import BeautifulSoup
import os
import json
import xml.etree.ElementTree as ET

def detect_file_type(file_content):
    """
    检测文件内容的类型
    返回: 'json', 'xml', 或 'html'
    """
    # 尝试解析为JSON
    try:
        json.loads(file_content)
        return 'json'
    except json.JSONDecodeError:
        pass
    
    # 尝试解析为XML
    try:
        ET.fromstring(file_content)
        return 'xml'
    except ET.ParseError:
        pass
    
    # 默认作为HTML处理
    return 'html'

def extract_titles_from_json(json_content):
    """从JSON内容中提取论文标题"""
    data = json.loads(json_content)
    titles = []
    
    if 'result' in data and 'hits' in data['result']:
        hits = data['result']['hits']['hit']
        for hit in hits:
            if 'info' in hit and 'title' in hit['info']:
                titles.append(hit['info']['title'])
    
    return titles

def extract_titles_from_xml(xml_content):
    """从XML内容中提取论文标题"""
    root = ET.fromstring(xml_content)
    titles = []
    
    # 查找所有title元素
    for hit in root.findall('.//hit'):
        title_elem = hit.find('.//title')
        if title_elem is not None and title_elem.text:
            titles.append(title_elem.text)
    
    return titles

def extract_titles_from_html(html_content):
    """从HTML内容中提取论文标题"""
    soup = BeautifulSoup(html_content, 'html.parser')
    titles = soup.find_all('span', class_='title')
    return [title.text.strip() for title in titles]

def extract_paper_titles(input_path, output_path=None):
    """
    从指定的会议文件中提取论文标题
    
    Args:
        input_path (str): 输入文件路径(HTML、JSON或XML)
        output_path (str, optional): 输出文件路径
    """
    try:
        # 如果没有指定输出路径,则基于输入文件名生成
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}.txt".replace('html', 'title')
        
        # 读取输入文件
        with open(input_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        # 检测文件类型并提取标题
        file_type = detect_file_type(file_content)
        print(f"**检测到{file_type.upper()}格式文件**")
        
        if file_type == 'json':
            titles = extract_titles_from_json(file_content)
        elif file_type == 'xml':
            titles = extract_titles_from_xml(file_content)
        else:  # html
            titles = extract_titles_from_html(file_content)
        
        # 将标题写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, title in enumerate(titles, 1):
                f.write(f"{i}. {title}\n")
        
        print(f"**成功提取 {len(titles)} 篇论文标题**")
        print(f"**已保存到: {output_path}**")
        
        return len(titles)
        
    except Exception as e:
        print(f"**处理过程中出现错误**: {str(e)}")
        return 0

def process_directory(input_dir):
    """
    处理指定目录下的所有HTML、JSON和XML文件
    
    Args:
        input_dir (str): 输入目录路径
    """
    try:
        total_files = 0
        total_papers = 0
        
        for filename in os.listdir(input_dir):
            if filename.endswith(('.html', '.json', '.xml')):
                input_path = os.path.join(input_dir, filename)
                papers_count = extract_paper_titles(input_path)
                total_files += 1
                total_papers += papers_count
        
        print(f"\n**处理统计**:")
        print(f"- 处理文件数: {total_files}")
        print(f"- 总论文数: {total_papers}")
        
    except Exception as e:
        print(f"**目录处理出错**: {str(e)}")

if __name__ == "__main__":
    # 单个文件处理示例
    # input_file = "conference_data.xml"  # 可以是.html, .json 或 .xml
    # if os.path.exists(input_file):
    #     extract_paper_titles(input_file)
    
    # 目录处理示例
    input_dir = "html"
    if os.path.exists(input_dir):
        process_directory(input_dir)

