CATEGORIES_CODES = {
    'alert': 0, 'button': 1, 'card': 2, 'checkbox_checked': 3,
    'checkbox_unchecked': 4, 'chip': 5, 'data_table': 6, 'dropdown_menu': 7,
    'floating_action_button': 8, 'grid_list': 9, 'image': 10, 'label': 11, 'menu': 12,
    'radio_button_checked': 13, 'radio_button_unchecked': 14, 'slider': 15, 'switch_disabled': 16,
    'switch_enabled': 17, 'text_area': 18, 'text_field': 19, 'tooltip': 20
}
CATEGORIES_REVERSE = {v: k for k, v in CATEGORIES_CODES.items()}

COUNT = CATEGORIES_CODES.__len__()
SHAPE = (224, 224, 1)

train_set_path = 'data/processed/train_set.csv'
test_set_path = 'data/processed/test_set.csv'
test_folder_path = 'data/test'
