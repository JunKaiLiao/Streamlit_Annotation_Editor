import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import json
from utils import getFirstKey, draw_mask, ANN_id, save2Local, SAVE2JSON

# from utils import draw_mask, ANN_id, SAVE2JSON
def make_sessionState(BACKUP2):
    if 'json' not in st.session_state:# used to save json info
        st.session_state['json'] = BACKUP2
#-------Initial Parameters---------------------------
if 'enterX' not in st.session_state:  # store the clicked points x, designed for save button
    st.session_state['enterX'] = []   
if 'enterY' not in st.session_state:  # store the clicked points y, designed for save button
    st.session_state['enterY'] = []
if 'oral' not in st.session_state:    # store the label, designed for save button
    st.session_state['oral'] = []
if 'switch' not in st.session_state:  # avoid overwriting the st.session_state['JsonInfo']
    st.session_state['switch'] = True
if 'ALL_keys' not in st.session_state:
    st.session_state['ALL_KEYS'] = []
if 'expand' not in st.session_state: # after uploaded all file, automatically roll up the expander
    st.session_state['expand'] = True
# if 'test' not in st.session_state:  # test flow
#     st.session_state['test'] = 0

put_id = []; ALL_KEYS=[]
temp_x=[];temp_y=[];temp_label=[]     # store current annotation info, help to update the dataframe
data={}                       
color2label = {'rgba(179, 255, 179, 0.5)':'G'
              ,'rgba(255, 255, 102, 0.5)':'Y'
              ,'rgba(255, 153, 153, 0.5)':'R'}
#--------Main Page-----------------------------------
st.set_page_config(     # 網頁標籤
    page_title="VIA Annotation Editting APP",
    page_icon="✏️",
    layout="wide",
)
st.title('✏️ VIA Annotation Editting APP') 
#----------------------------------------------------
st.sidebar.title('⚙️Functions')
with st.sidebar.expander('🗂️ Upload files', expanded=st.session_state['expand']):
    uploaded_json = st.file_uploader("📝 Select an VIA json file"
                         , type=['json']
                         , accept_multiple_files=False)
   
    uploaded_img = st.file_uploader("🖼️ Select Original image"
                         , type=['png', 'jpg', 'jpeg']
                         , accept_multiple_files=True)
    
    uploaded_pred = st.file_uploader("🖼️ Select an predicted image"
                         , type=['png', 'jpg', 'jpeg']
                         , accept_multiple_files=True)
   
#-------selectbox---------------------
if uploaded_img is not None:
    tail_list =[]; head_list = []
    for i in range(len(uploaded_img)):
        head, sep, tail = str(uploaded_img[i].name).partition(".") # tail-> file type, head -> file name
        head_list.append(head)
        tail_list.append(tail)

if uploaded_pred is not None:
   pred_tail_list =[]; pred_head_list = []
   for i in range(len(uploaded_pred)):
      head, sep, tail = str(uploaded_pred[i].name).partition(".") # tail-> file type, head -> file name
      pred_head_list.append(head)
      pred_tail_list.append(tail)

select_img = st.sidebar.selectbox('Please select a image', options=head_list, label_visibility="hidden")    
st.sidebar.header("✍️ADD")
label = st.sidebar.radio("Please select a class.",
                        ('Green', 'Yellow', 'Red'))

if label == 'Green':
    stroke_color = '#00ff00'
    fill_color = 'rgba(179, 255, 179, 0.5)'
elif label == 'Yellow':
    stroke_color = '#ffff00'
    fill_color = 'rgba(255, 255, 102, 0.5)'
else: # Red
    stroke_color = '#ff3300'
    fill_color = 'rgba(255, 153, 153, 0.5)'

#---------Load Json--------------------
if uploaded_json is not None:  # successfully load json, yet save and export have not been done
   BACKUP = pd.read_json(uploaded_json)
   for key in BACKUP.keys():
        ALL_KEYS.append(key)
   if st.session_state['switch']:
        make_sessionState(BACKUP) # make 'json' session_state to store Info before rerun
        st.session_state['switch'] = False
#---------Image Display-------------------
img_col1, img_col2  = st.columns(2)
with img_col1:
    if pred_head_list and uploaded_pred is not None:
        pred_index = pred_head_list.index(select_img)         
        pred_img = Image.open(uploaded_pred[pred_index])
        st.header('Yolo-v7 Predicted Image')
        st.image(pred_img, width=480)
        pred_img.close()
        if st.session_state['expand']:
            st.session_state['expand'] = False
            st.experimental_rerun()

if uploaded_img is not None and head_list and uploaded_json is not None and select_img and uploaded_pred is not None and ALL_KEYS:
    ori_index = head_list.index(select_img)
    img = Image.open(uploaded_img[ori_index])
    filename = head_list[ori_index]+'.'+tail_list[ori_index]
    firstkey = getFirstKey(filename, ALL_KEYS)
    save2Local(img, filename)
    draw_mask(st.session_state['json'], firstkey, filename)
    width, height = img.size
    wRatio = width / 480
    hRatio = height / 360
    with img_col2: 
        st.header('Ground Truth Image')
        canvas_result = st_canvas(
                        fill_color=fill_color,  # Fixed fill color with some opacity
                        stroke_width= 2 ,       # 畫筆粗度
                        stroke_color=stroke_color, # 畫筆顏色
                        background_color='#ffffff',# 圖片背景為白色
                        background_image=Image.open('mask.png') if uploaded_img else None,
                        update_streamlit=True,     # 即時更新標註
                        height=360,
                        width= 480,
                        drawing_mode='polygon',
                        point_display_radius=2,
                        key="canvas",
                        )
    
        #------------store annotation info---------------------------------------
        if canvas_result.json_data is not None:
            INFO = canvas_result.json_data['objects']
            for index in range(len(INFO)):
                num_points = len(INFO[index]['path']) - 1 # 最後一個點不計
                X=[];Y=[]
                for point in range(num_points):
                    X.append(round(float(INFO[index]['path'][point][1])*wRatio))
                    Y.append(round(float(INFO[index]['path'][point][2])*hRatio))
                temp_label.append(color2label[INFO[index]['fill']]) # 藉由標註顏色取得類別
                temp_x.append(X)
                temp_y.append(Y)
            #------------make & display dataframe-------------------------------
            data = {'type':temp_label
                    ,'X'  :temp_x
                    ,'Y'  :temp_y }
    put_id = ANN_id(st.session_state['json'], firstkey)
# -------------------------------------Delete Annotation-------------------------
with st.sidebar.form('DELETE'):
    st.header('❌Delete')
    options = st.multiselect('Keep or remove the annotations from VIA json file', put_id, put_id)
    del_list = [item for item in put_id if item not in options]
    renew = st.form_submit_button('Delete')
    if renew:
        for item in del_list:
            pop_index = put_id.index(item)
            st.session_state['json'][firstkey]['regions'].pop(pop_index)
            del put_id[pop_index]
        st.experimental_rerun()

# ----------Reset & Save & Export------------------------------------------------------
with st.sidebar:
    scol1, scol2, scol3 = st.sidebar.columns([3,3,5])
    with scol1:
        reset_del = st.button('Reset')
        if reset_del:  # 若按下重置按鈕，將被刪除的資訊用備份的資料補起來 
            st.session_state['json'][firstkey]['regions'] = BACKUP[firstkey]['regions']
            st.write(BACKUP[firstkey]['regions'])
            st.experimental_rerun()
    with scol2:
        save = st.button('SAVE')
    with scol3:
        export = st.button('EXPORT')


placeholder = st.empty()    # 以解決dataframe 重複出現的問題
if data:
    placeholder.empty() # clear previous dataframe
    placeholder.dataframe(pd.DataFrame(data), use_container_width=True)


# ----------------imcomplete--------------------------------------------------
if save: # 當按下儲存，將目前資料存到jsonInfo裡，並將st.session 'enterX' 'enterY' 'oral'變為初始狀態(empty list), 跳下一張影像
    INFO = canvas_result.json_data['objects']
    for index in range(len(INFO)):
        num_points = len(INFO[index]['path']) - 1 # 最後一個點不計
        X=[];Y=[]
        for point in range(num_points):
            X.append(round(float(INFO[index]['path'][point][1])*wRatio))
            Y.append(round(float(INFO[index]['path'][point][2])*hRatio))
        st.session_state['oral'].append(color2label[INFO[index]['fill']]) # 藉由標註顏色取得類別
        st.session_state['enterX'].append(X)
        st.session_state['enterY'].append(Y)
    st.session_state['json'] = SAVE2JSON(st.session_state['json'], st.session_state['oral'] # 必須考量到刪除功能
                                        ,st.session_state['enterX'], st.session_state['enterY']
                                        ,firstkey, filename) 
    
    # clear the space to store next image
    st.session_state['oral'] = []; st.session_state['enterX']=[]; st.session_state['enterY']
    st.experimental_rerun()

# if export: # 當按下匯出，將st.session_state['JsonInfo']存成json檔，並下載到使用者的電腦上

st.sidebar.write('Made by Jun-Lai Liao 🤗')



#----------metric---------------------
    # mcol1, mcol2 = st.columns(2)
    # with mcol1:
    #     st.metric('Width', str(width))
    # with mcol2:
    #     st.metric('Height', str(height))