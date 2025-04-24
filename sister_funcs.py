import pandas as pd,pylab as pl,numpy as np,sys,pickle,io,lightgbm as lgb,pyspark,openai,base64,os,json
from google import genai
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

def match_images(url1,url2,client,model='gemma3:12b',debug=False,use_gemini=True):


     if use_gemini:
          with open(url1, "rb") as f1, open(url2, "rb") as f2:
               img_bytes1 = f1.read()
               img_bytes2 = f2.read()
     else:
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

          if use_gemini:
               response=client.models.generate_content(
                    model='gemini-2.5-flash-preview-04-17',
                    contents=[
                         the_prompt,
                         genai.types.Part.from_bytes(data=img_bytes1, mime_type="image/jpeg"),
                         genai.types.Part.from_bytes(data=img_bytes2, mime_type="image/jpeg")
                    ],
                    config=types.GenerateContentConfig(
                         thinking_config=types.ThinkingConfig(thinking_budget=0)
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



if __name__=='__main__':

     uspics=glob("us/*_50.jpg")
     ugpics=glob("uganda/*_50.jpg")
     #uspics=glob("serene_us_batch/*_15.JPG")
     #ugpics=glob("serene_uganda_batch/*_15.JPG")
     
     #gclient=OpenAI(base_url=os.environ['GEMINIURL'],api_key=os.environ['GEMINIKEY'])
     gclient = genai.Client(api_key=os.environ['GEMINIKEY'])

     matches=[]
     
     for pics in tqdm([(uspic,ugpic) for uspic in uspics for ugpic in ugpics][:2]):
         matches.append(match_images(*pics,gclient,GEMINI_FLASH_MODEL))

     sispf=pd.DataFrame(matches)
                        
