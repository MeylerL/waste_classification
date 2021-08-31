import pandas as pd
import json
import shutil

# dict converting taco numbered categories into trashnet string categories
category_conversion = {}
category_conversion['metal'] = [0,8,10,11,12,28]
category_conversion['cardboard'] = [13,14,15,16,17,18,19,20]
category_conversion['glass'] = [6,9,23,26]
category_conversion['paper'] = [21,30,31,32,33,34]
category_conversion['plastic'] = [4,5,7,24,27,43,44,47,49,55]
category_conversion['trash'] = [1,2,3,22,25,29,35,36,37,38,39,40,41,42,45,46,
                                48,50,51,52,53,54,57,58,59]


def move_rename_jsons():
    """moves the jsons into one directory and renames them by batch"""
    # you will need to choose or create a directory for second part of shutil
    for i in range(1,16):
         shutil.move(f'<YOUR/PATH/HERE>/TACO/data/batch_{i}/annotations.json',
                f'<YOUR/PATH/HERE>/batch_{i}_annotations.json')
    return None

def open_json(n):
    """opens the json for taco batch and extracts the image and category data"""
    file = open(f'<YOUR/PATH/HERE>/batch_{n}_annotations.json')
    data = json.load(file)
    image_data = data['images']
    category_data = data['annotations']
    return image_data, category_data

def image_ids(image_data):
    """ returns a dict with image_ids and image file names"""
    image_files = {}
    for item in image_data:
        image_files[item['id']] = item['file_name']
    return image_files

def image_categories(category_data):
    """ returns a dict with the image_ids and the taco category numbers
    labelling that image"""
    image_categories_dict = {}
    for item in category_data:
        if item['image_id'] in image_categories_dict:
            image_categories_dict[item['image_id']].append(item['category_id'])
        else:
            image_categories_dict[item['image_id']] = [item['category_id']]
    for key in image_categories_dict:
        image_categories_dict[key] = list(set(image_categories_dict[key]))
    return image_categories_dict

def compatible_images(image_categories):
    """ returns a dict with image_id and category, only includes images that
    are labelled with one trashnet category"""
    new_image_categories = {}
    for key in image_categories.keys():
        target_list = []
        for label in image_categories[key]:
            if label in category_conversion['trash']:
                target_list.append('trash')
            elif label in category_conversion['cardboard']:
                target_list.append('cardboard')
            elif label in category_conversion['plastic']:
                target_list.append('plastic')
            elif label in category_conversion['paper']:
                target_list.append('paper')
            elif label in category_conversion['metal']:
                target_list.append('metal')
            elif label in category_conversion['glass']:
                target_list.append('glass')
        new_image_categories[key] = target_list
    for key in new_image_categories:
        new_image_categories[key] = list(set(new_image_categories[key]))
    final_images = {k:v for k,v in new_image_categories.items() if len(v) <= 1}
    return final_images

def move_images(n, final_images, image_files):
    """creates dataframe of images to be moved and moves them to relevant
    trashnet folder"""
    id_category_df = pd.DataFrame.from_dict(final_images, orient ='index')
    id_filename_df = pd.DataFrame.from_dict(image_files, orient ='index')
    image_df = id_category_df.merge(id_filename_df, left_index=True, right_index=True)
    image_df.rename(columns = {'0_x':'category', '0_y':'filename'}, inplace = True)
    # need to create trashnet folders that match the category names for second part of shutil
    for index, row in image_df.iterrows():
        shutil.move(f'<YOUR/PATH/HERE>/TACO/data/batch_{n}/{row["filename"]}',
                f'<YOUR/PATH/HERE>/{row["category"]}/batch_{n}{row["filename"]}')
    return None

def get_tacos():
    """loops through all jsons and applies the above functions in order to unpack"""
    for item in range(1,16):
        image_data, category_data = open_json(item)
        image_files = image_ids(image_data)
        image_cats = image_categories(category_data)
        final_images = compatible_images(image_cats)
        move_images(item, final_images, image_files)
    return None
