import pandas as pd,pylab as pl,numpy as np,sys,pickle,io,lightgbm as lgb,pyspark,openai,base64,os,json,re
from google import genai
from json_repair import repair_json
from tqdm import tqdm
from glob import glob
from openai import OpenAI

GEMINI_FLASH_MODEL='gemini-2.5-flash-preview-04-17'
GEMINI_PRO_MODEL='gemini-2.5-pro-preview-03-25'
QWEN_VL_MODEL='ZimaBlueAI/Qwen2-VL-7B-Instruct:latest'

def csv2html(csv_paths,out_path="image_pairs.html",shuffle=True):

     import pandas as pd

     # Read the CSV files
     # If csv_path is a dataframe then just use that
     df = csv_paths if type(csv_paths)==pd.DataFrame else pd.concat([pd.read_csv(csv_path).loc[lambda x:x.matching==True] for csv_path in csv_paths],axis=0)

     if shuffle: df=df.sample(frac=1)
     
     # Generate HTML content
     html_content = '<html><head><title>Image Pairs</title></head><body>'
     html_content += '<table border="1" style="width:100%; text-align:center;">'
     html_content += '<tr><th>Image 1</th><th>Image 2</th><th>Matching</th></tr>'

     for _, row in df.iterrows():
          html_content += '<tr>'
          html_content += f'<td><img src="{row["image1"]}" alt="Image 1" style="max-width:500px; max-height:500px;"></td>'
          html_content += f'<td><img src="{row["image2"]}" alt="Image 2" style="max-width:500px; max-height:500px;"></td>'
          html_content += f'<td>{row["matching"]}</td>'
          html_content += '</tr>'

     html_content += '</table></body></html>'

     # Write the HTML content to a file
     with open(out_path, 'w') as f:
          f.write(html_content)


def subsample_all_images(imlist,scale_pct=15):

     for im in tqdm(imlist):
          impath=os.path.splitext(im)
          os.system(f'magick "{im}" -scale {scale_pct}% "{impath[0]}_{scale_pct}{impath[1]}"')

def match_images(url1,url2,client,model='gemma3:12b',debug=False,use_gemini=True,really_tough=False):


     if use_gemini:
          with open(url1, "rb") as f1, open(url2, "rb") as f2:
               img_bytes1 = f1.read()
               img_bytes2 = f2.read()
     else:
          imdat1=base64.b64encode(open(url1,"rb").read()).decode("utf-8")
          imdat2=base64.b64encode(open(url2,"rb").read()).decode("utf-8")

     
     the_prompt=f"""
1. Please examine the two images provided.
2. Determine if the objects being held by the individuals in each image are exactly the same (for example, the same brand, type, appearance).
3. Respond only with a JSON object in the following format:
{{
  "image1": "{url1}",
  "image2": "{url2}",
  "matching": <true/false>
}}
Only set "matching": true if the objects are clearly and unmistakably exactly the same.
Do not include any explanations or comments—return just the JSON object.  
"""
     if really_tough:
          the_prompt=f"""
1. Please examine the two images provided.
2. Determine if the objects being held by the individuals in each image are exactly the same - it has to be EXACTLY the same, i.e. same brand, model, design, appearance, color etc.
3. Respond only with a JSON object in the following format:
{{
  "image1": "{url1}",
  "image2": "{url2}",
  "matching": <true/false>
}}
Only set "matching": true if the objects are clearly and unmistakably exactly the same.
Do not include any explanations or comments—return just the JSON object.  
"""
     
     if debug: print(the_prompt)
     
     try:

          if use_gemini:
               response=client.models.generate_content(
                    model=model,
                    contents=[
                         the_prompt,
                         genai.types.Part.from_bytes(data=img_bytes1, mime_type="image/jpeg"),
                         genai.types.Part.from_bytes(data=img_bytes2, mime_type="image/jpeg")
                    ],
                    config=genai.types.GenerateContentConfig(
                         thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
                    )  # This disables thinking mode
               )
               res=response.candidates[0].content.parts[0].text
               
          else:     
               response=client.chat.completions.create(model=model,
                         messages=[{'role':'user','content':[{'type':'text','text':the_prompt},
                         {'type':'image_url','image_url':{'url':f"data:image/jpeg;base64,{imdat1}"}},
                         {'type':'image_url','image_url':{'url':f"data:image/jpeg;base64,{imdat2}"}}
                                                    ]}])
               res=response.choices[0].message.content
         
          return json.loads(repair_json(res))

     except:

         return {'image1':url1,'image2':url2,'matching':'error'}

def rename_and_copy_files(filepf,new_dir,filename_prefix='image',remove_regex=''):

     counter=iter(range(filepf.shape[0]+1))
     reg=re.compile(remove_regex)
     
     def rename_file(x,y):
          i=next(counter)
          return f"cp {reg.sub('',x)} {new_dir}/{filename_prefix}_{i}a{os.path.splitext(x)[1].lower()} ; "+\
               f"cp {reg.sub('',y)} {new_dir}/{filename_prefix}_{i}b{os.path.splitext(y)[1].lower()}"
         
     filepf=filepf.copy()               
     filepf['cmd']=[rename_file(*tmp) for tmp in zip(filepf.image1,filepf.image2)]

     return filepf

if __name__=='__main__':

     #uspics=glob("us/*_50.jpg")
     #ugpics=glob("uganda/*_50.jpg")
     uspics=glob("serene_us_batch/*_15.JPG")
     ugpics=glob("serene_uganda_batch/*_15.JPG")
     
     #gclient=OpenAI(base_url=os.environ['GEMINIURL'],api_key=os.environ['GEMINIKEY'])
     gclient = genai.Client(api_key=os.environ['GEMINIKEY'])

     full_set=[(uspic,ugpic) for uspic in uspics for ugpic in ugpics]
     chunks=list(range(0,len(full_set),300))+[len(full_set)]

     for chunk in range(1,len(chunks)):
     
          print(f"Getting chunk #{chunk} out of {len(chunks)}")

          matches=[]
          chunk_ims=full_set[chunks[chunk-1]:chunks[chunk]]
          
          for pics in tqdm(chunk_ims):
               matches.append(match_images(*pics,gclient,GEMINI_FLASH_MODEL))

          sispf=pd.DataFrame(matches)
          sispf.to_csv(f"sispf_{chunk}.csv",index=False)

     # Code to refine using pplx api

     # matches=[]
     # for _,row in tqdm(foo.iterrows()):
     #      matches.append(match_images(row.image1,row.image2,pclient,model='sonar',use_gemini=False))
