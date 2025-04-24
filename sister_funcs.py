import pandas as pd,pylab as pl,numpy as np,sys,pickle,io,lightgbm as lgb,pyspark,openai,base64,os,json
from json_repair import repair_json
from tqdm import tqdm
from glob import glob
from openai import OpenAI

GEMINI_FLASH_MODEL='gemini-2.5-flash-preview-04-17'
GEMINI_PRO_MODEL='gemini-2.5-pro-preview-03-25'
GEMMA_3_MODEL='gemma-3-27b-it'

def subsample_all_images(imlist,scale_pct=15):

     for im in tqdm(imlist):
          impath=os.path.splitext(im)
          os.system(f'magick "{im}" -scale {scale_pct}% "{impath[0]}_{scale_pct}{impath[1]}"')

def match_images(url1,url2,client,model='gemma3:12b',debug=False):
    
     imdat1=base64.b64encode(open(url1,"rb").read()).decode("utf-8")
     imdat2=base64.b64encode(open(url2,"rb").read()).decode("utf-8")

     the_prompt=f"""
1. Please examine the two images provided.
2. Determine if the objects being held by the individuals in each image are exactly the same (for example, the same brand, type, appearance and/or model of school product).
3. Respond only with a JSON object in the following format:
{{
  "image1": "{url1}",
  "image2": "{url2}",
  "matching": <true/false>
}}
Only set "matching": true if the objects are clearly and unmistakably the same.
Do not include any explanations or comments—return just the JSON object.  
"""
     if debug: print(the_prompt)
     
     try:
         res=client.chat.completions.create(model=model,
                         messages=[{'role':'user','content':[{'type':'text','text':the_prompt},
                         {'type':'image_url','image_url':{'url':f"data:image/jpeg;base64,{imdat1}"}},
                         {'type':'image_url','image_url':{'url':f"data:image/jpeg;base64,{imdat2}"}}
                                                    ]}])
         
         return json.loads(repair_json(res.choices[0].message.content))

     except:

         return {'image1':url1,'image2':url2,'matching':'error'}



if __name__=='__main__':

     uspics=glob("us/*_50.jpg")
     ugpics=glob("uganda/*_50.jpg")

     gclient=OpenAI(base_url=os.environ['GEMINIURL'],api_key=os.environ['GEMINIKEY'])

     matches=[]
     
     for pics in tqdm([(uspic,ugpic) for uspic in uspics for ugpic in ugpics]):
         matches.append(match_images(*pics,gclient,GEMINI_FLASH_MODEL))

     sispf=pd.DataFrame(matches)
                        
