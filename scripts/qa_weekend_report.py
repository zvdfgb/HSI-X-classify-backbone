"""Render the exported PDF and audit values. Human image review still required."""
import argparse,json,re
from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image,ImageOps,ImageDraw
from docx import Document

TITLE='TransMixer 周末融合与知识蒸馏实验全面报告'

def main(folder):
    data=json.loads((folder/'周末实验总汇.json').read_text(encoding='utf-8'))
    pdf=PdfReader(folder/(TITLE+'.pdf'));qa=folder/'页面核验';qa.mkdir(exist_ok=True)
    page_text=[page.extract_text() for page in pdf.pages];text='\n'.join(page_text)
    checks={f"{r['id']}_OA":f"{r['oa']:.4f}" in text for r in data['legacy_methods']}
    checks.update({f"{r['id']}_AA":f"{r['aa']:.4f}" in text for r in data['legacy_methods']})
    checks.update({f"{r['id']}_Kappa":f"{r['kappa']:.4f}" in text for r in data['legacy_methods']})
    doc=Document(folder/(TITLE+'.docx'))
    md=(folder/(TITLE+'.md')).read_text(encoding='utf-8')
    checks['no_inline_dollar_math']=not bool(re.search(r'\$[^$\n]+\$',md))
    checks['all_pages_have_text']=all(len(t)>100 for t in page_text)
    checks['expected_twelve_pages']=len(pdf.pages)==12
    checks['title_is_title_style']=doc.paragraphs[0].style.name=='Title'
    images=convert_from_path(folder/(TITLE+'.pdf'),dpi=130,poppler_path='C:/Users/mmm/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
    for i,im in enumerate(images):
        im.save(qa/f'page-{i+1:02d}.png');(qa/f'page-{i+1:02d}.txt').write_text(page_text[i],encoding='utf-8')
    contact=Image.new('RGB',(1110,540*((len(images)+2)//3)),'#dddddd');draw=ImageDraw.Draw(contact)
    for i,im in enumerate(images):
        thumb=ImageOps.contain(im,(350,500));x=i%3*370+10;y=i//3*540+25
        contact.paste(thumb,(x,y));draw.text((x,y-18),f'Page {i+1}',fill='black')
    contact.save(qa/'contact.png')
    report={'pages':len(pdf.pages),'checks':checks,'passed':all(checks.values()),'page_text':[{'page':i+1,'characters':len(t),'start':t[:120],'end':t[-100:]} for i,t in enumerate(page_text)],'renderer':'Word or canonical DOCX renderer to PDF; Poppler 130 dpi PNG','visual_review':'PENDING agent must inspect every page PNG before delivery'}
    (qa/'核验记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pages':len(pdf.pages),'checks_passed':all(checks.values()),'failed':[k for k,v in checks.items() if not v],'contact':str(qa/'contact.png')},ensure_ascii=False))
    if not all(checks.values()):raise SystemExit('Report checks failed; repair layout or content and render again')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);main(p.parse_args().run.resolve())
