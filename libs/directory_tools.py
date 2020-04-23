import os

def create_tree( level_list, init_dir = "." ) :
    if len(level_list) > 1:
    
        for new_dir in level_list[0]:

            os.mkdir(init_dir+'/'+new_dir)
            if len(level_list) >=2:
                create_tree(level_list[1:] , init_dir+'/'+new_dir)
    else:
        for new_dir in level_list[0]:

            os.mkdir(init_dir+'/'+new_dir)


