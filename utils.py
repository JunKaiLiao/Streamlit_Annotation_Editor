import cv2
import numpy as np
import os

def draw_mask(jsonInfo, firstKey, filename):
    ori_img = cv2.imread(filename) 
    height , width, channel = ori_img.shape
    ori_img_clone = ori_img.copy()
    num_mask = len(jsonInfo[firstKey]['regions'])
    id = 1
    color_dict = {'G':(0, 255, 0), 'R':(0, 0, 255), 'Y':(0, 128, 255)}

    if num_mask == 0:  # 當刪除所有標註，顯示原圖
        cv2.imwrite('mask.png', ori_img_clone)

    if height > 1900:
        line_width = 10
        font_size = 10
    else:
        line_width = 1
        font_size = 1

    for i in range(num_mask):
        position=[]
        x = jsonInfo[firstKey]['regions'][i]['shape_attributes']['all_points_x']
        y = jsonInfo[firstKey]['regions'][i]['shape_attributes']['all_points_y']
        # caluate the mean point to put label id
        id_position = (int(np.mean(x)), int(np.mean(y)))
        label = jsonInfo[firstKey]['regions'][i]['region_attributes']['oral']
            # print(x)
            # print(y)
            # print(label)
        for j in range(len(x)):
            position.append([x[j], y[j]])
        position_array = np.array([position])
        ori_img = cv2.polylines(ori_img, [position_array], isClosed=True
                                , color=color_dict[label], thickness=line_width)
        cv2.putText(ori_img, str(id), id_position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color_dict[label],4, cv2.LINE_AA); id+=1
        cv2.imwrite('mask.png', ori_img)

def ANN_id(jsonInfo, firstKey):
    put_id=[]; id=1; 
    num = len(jsonInfo[firstKey]['regions'])
    for i in range(len(jsonInfo[firstKey]['regions'])):  # 若num = 0, 不會進迴圈
        label = jsonInfo[firstKey]['regions'][i]['region_attributes']['oral']
        put_id.append(str(id)+'. '+label)
        id += 1 
    return put_id

def getFirstKey(filename, ALL_KEYS): # use part of an element to get the rest of the part of an element
    for key in ALL_KEYS:
        if filename in key:
            key_index = ALL_KEYS.index(key)
            return ALL_KEYS[key_index]

def save2Local(img, filename):
    img_array = np.array(img)
    cv2.imwrite(filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    img.close()

def SAVE2JSON(jsonInfo, label, enterX, enterY, firstkey, filename):
    save_data = []
    rest_of_mask = len(jsonInfo[firstkey]['regions'])   # 將沒被刪除的mask提取出來
    for i in range(rest_of_mask):
        enterX.append(jsonInfo[firstkey]['regions'][i]['shape_attributes']['all_points_x'])
        enterY.append(jsonInfo[firstkey]['regions'][i]['shape_attributes']['all_points_y'])
        label.append(jsonInfo[firstkey]['regions'][i]['region_attributes']['oral'])

    for i in range(len(label)):
        save_data.append({"shape_attributes":{"name":"polygon",
                                              "all_points_x": enterX[i],
                                              "all_points_y": enterY[i]},
                          "region_attributes":{"oral":label[i]}}
                        )
    jsonInfo[firstkey]['regions'] = save_data
    jsonInfo.to_json('./test.json')
    os.remove(filename)
    # with open('./test.json', 'w') as complete:
    #     json.dump(jsonInfo, complete)
    return jsonInfo