import os
from ipywidgets import widgets, Layout, HBox, VBox
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import threading
import random

import degradation_utils as hp


class RegDataFilterUI:

    def __init__(self, img_dir, mask_dir, annot_dir, output_dir, save_checkpoint_file=None):

        # set data loading info
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.annot_dir = annot_dir
        self.output_dir = output_dir
        self.random_flag = False

        # load image names
        self.img_list = sorted(os.listdir(img_dir))

        # txt file to save filter images name
        if save_checkpoint_file is not None:
            self.filtered_txt_file = save_checkpoint_file
        else:
            self.filtered_txt_file = os.path.join(output_dir, 'filtered_img_names.txt')

        saved_filenames = []
        if os.path.exists(self.filtered_txt_file):     # check if same file already exists:
            with open(self.filtered_txt_file, 'r') as file:
                saved_filenames = file.read()
                saved_filenames = saved_filenames.split()
        self.save_counter = len(saved_filenames)

        # ensure all necessary folders are available
        os.makedirs(output_dir, exist_ok=True)

        # additional parameters
        self.pos_counter = 0
        self.cache = {}

        # set up initial UI
        self.pos_counter = 0
        filename = self.img_list[self.pos_counter]
        img, annot_img, mask, annot_data = self.load_data(filename)

        # save current variables for easy access
        self.cache["original_img"] = img
        self.cache["annot_img"] = annot_img
        self.cache["annot_data"] = annot_data

        # set initial UI
        self.init_ui_elements(img=annot_img, mask=mask, annot_data=annot_data)

    def init_ui_elements(self, img, mask, annot_data):

        # --- Label ---
        self.fn_label = widgets.Label(value=f"Filename: {annot_data['image']}",
                                      layout=Layout(padding='0', margin='0 0 0 0'))
        self.saved_cnt_label = widgets.Label(value=f"Total saved images: {self.save_counter}",
                                             layout=Layout(padding='0', margin='0 0 0 0'))

        # --- Images (aligned tightly) ---
        self.image_output = widgets.Output(layout=Layout(flex='2', padding='0', margin='0'))
        self.mask_output = widgets.Output(layout=Layout(flex='1', padding='0', margin='0'))
        self.update_images(img=img, mask=mask)

        self.img_row = HBox([self.image_output, self.mask_output],
                            layout=Layout(padding='0', margin='0', width='100%', display='flex'))

        # --- Checkboxes ---
        checkboxes = self.generate_degradation_checkboxes(annot_data["annotations"])
        self.ch_label = widgets.Label(value=f"Segments to keep: ",
                                      layout=Layout(padding='0', margin='0 0 5px 0'))

        self.chbox_row = HBox(
            children=[self.ch_label] + checkboxes,
            layout=Layout(
                overflow_x='auto',  # Enable horizontal scrolling if needed
                overflow_y='hidden',  # Disable vertical scrolling
                flex_flow='row nowrap',
                height="33px",
                padding='0',
                margin='0',
                display='flex',
            )
        )

        # --- Navigation buttons ---
        self.prev_button = widgets.Button(description='Prev')
        self.next_button = widgets.Button(description='Next')
        self.save_button = widgets.Button(description='Save')
        self.toggle_btn = widgets.ToggleButton(
            value=False,
            description='Hide bbox',
            disabled=False,
            button_style='',
            icon='eye-slash'
        )
        self.toggle_rand_btn = widgets.ToggleButton(
            value=False,
            description='Not Random',
            disabled=False,
            button_style='',
            icon='xmark'
        )
        self.msg_output = widgets.Output()

        # Set up the button click event
        self.next_button.on_click(lambda b: self.load_new_image(b, increment=1))
        self.prev_button.on_click(lambda b: self.load_new_image(b, increment=-1))
        self.toggle_btn.observe(self.on_toggle_click, names='value')
        self.toggle_rand_btn.observe(self.on_toggle_random_click, names="value")
        self.save_button.on_click(self.on_save_click)

        self.button_row = HBox(
            children=[self.toggle_rand_btn, self.prev_button, self.next_button, self.toggle_btn, self.save_button,
                      self.msg_output],
            layout=Layout(
                padding='0',
                margin='10px',
                display='flex',
            )
        )

        # --- Final GUI ---
        self.gui = VBox(
            children=[self.img_row, self.saved_cnt_label, self.fn_label, self.chbox_row, self.button_row],
            layout=Layout(
                width="80%",
                padding='0',
                margin='0',
                border='0')
        )

    def on_save_click(self, btn):

        # fetch the checkbox values that are currently selected
        checkboxes = self.chbox_row.children[1:]
        to_keep = []
        for c in checkboxes:
            if c.value:
                idx = int(c.description.split(" --> ", maxsplit=1)[0])
                to_keep.append(idx)

        if len(to_keep) == 0:
            self.show_message(msg="No annot to save", duration=2)  # display saved notification for 2 sec
            return

        # create a new annotation file with only degradation value accepted by the user
        # set other values to -1
        new_annot_data = self.cache["annot_data"].copy()
        updated_annot = []
        for component in new_annot_data["annotations"]:
            new_value = component["degradation"] if component["id"] in to_keep else -1
            component["manual_check"] = new_value
            updated_annot.append(component.copy())
        new_annot_data["annotations"] = updated_annot

        # save the annotation file
        filename = new_annot_data['image']
        file_initials = ".".join(filename.split(".")[:-1])  # remove extension
        json_out_path = os.path.join(self.output_dir, f"{file_initials}.json")
        with open(json_out_path, 'w') as jspot:
            json.dump(new_annot_data, jspot)

        # save the file name to txt file
        with open(self.filtered_txt_file, 'a+') as file:
            file.seek(0)  # Move the cursor to the beginning of the file to read
            file_content = file.read()
            if filename not in file_content:  # add filename if not present
                self.save_counter += 1
                file.write(filename + '\n')
            file.close()

        self.saved_cnt_label.value = f"Total saved images: {self.save_counter}"  # update counter label
        self.show_message(duration=2)   # display saved notification for 2 sec

    def show_message(self, msg="Saved!", duration=2):
        """Display a message for `duration` seconds"""
        with self.msg_output:
            clear_output(wait=True)  # Clear previous messages
            if msg == "Saved!":
                print(f"✅ \033[32m{msg}\033[0m")
            else:
                print(f"{msg}")

        def clear_msg():   # Schedule message removal
            with self.msg_output:
                clear_output()

        # Use thread to avoid blocking the notebook
        timer = threading.Timer(duration, clear_msg)
        timer.start()

    def on_toggle_click(self, change):

        if change['new']:  # When button is toggled on
            self.toggle_btn.description = 'View bbox'
            self.toggle_btn.icon = 'eye'

            # update with original image
            self.update_images(img=self.cache["original_img"], mask=None)


        else:  # When button is toggled off
            self.toggle_btn.description = 'Hide bbox'
            self.toggle_btn.icon = 'eye-slash'

            # update with annotated image
            self.update_images(img=self.cache["annot_img"], mask=None)

    def on_toggle_random_click(self, change):

        if change['new']:  # When button is toggled on
            self.toggle_rand_btn.description = 'Random'
            self.toggle_rand_btn.icon = 'check'
            self.random_flag = True
            self.prev_button.disabled = True

        else:  # When button is toggled off
            self.toggle_rand_btn.description = 'Not Random'
            self.toggle_rand_btn.icon = 'xmark'
            self.random_flag = False
            self.prev_button.disabled = False

    def update_images(self, img, mask):

        with self.image_output:
            clear_output(wait=True)
            plt.imshow(img)
            plt.axis("off")
            plt.title("Degradation values")
            plt.tight_layout(pad=0)
            plt.show()

        if mask is not None:
            with self.mask_output:
                clear_output(wait=True)
                plt.imshow(mask, cmap="gray")
                plt.axis("off")
                plt.title("Ground truth segmentation mask")
                plt.tight_layout(pad=0)
                plt.show()

    def generate_degradation_checkboxes(self, annot_data):
        chbox_labels = self.get_degradation_values(annot_data)
        checkboxes = [
            widgets.Checkbox(
                value=True,
                description=label,
                indent=False,
                layout=Layout(margin='0 5px 0 5px', padding='2px', width='auto')
            )
            for label in chbox_labels
        ]
        return checkboxes

    def load_new_image(self, b, increment=1):

        if self.random_flag:
            filename = random.choice(self.img_list)
            self.pos_counter = self.img_list.index(filename)
        else:
            self.pos_counter = (self.pos_counter + increment) % len(self.img_list)  # wrap around
            filename = self.img_list[self.pos_counter]

        img, annot_img, mask, annot_data = self.load_data(filename)
        self.cache["original_img"] = img
        self.cache["annot_img"] = annot_img
        self.cache["annot_data"] = annot_data

        self.fn_label.value = f"Filename: {annot_data['image']}"   # update filename label
        self.update_images(img=annot_img, mask=mask)               # update images

        # update checkbox
        new_checkboxes = self.generate_degradation_checkboxes(annot_data["annotations"])
        self.chbox_row.children = [self.ch_label] + new_checkboxes

        # reset the toggle button
        self.toggle_btn.value = False
        self.toggle_btn.description = 'hide bbox'
        self.toggle_btn.icon = 'eye-slash'

    def display_UI(self):
        display(self.gui)

    def get_degradation_values(self, annot_data):
        values = []
        for component in annot_data:
            ratio = component["degradation"]
            idx = component["id"]
            if ratio < 0:
                continue
            label = f"{idx} --> {round(ratio, 3)} "
            values.append(label)
        return values

    def load_data(self, img_name):

        mask_name = ".".join(img_name.split(".")[:-1]) + ".png"
        annot_name = ".".join(img_name.split(".")[:-1]) + ".json"

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        annot_path = os.path.join(self.annot_dir, annot_name)

        # load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load mask
        mask_img = cv2.imread(mask_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        mask_img = mask_img / 255
        mask_img = mask_img.astype(np.uint8)

        # load annotations
        with open(annot_path, 'r') as file:
            annot_data = json.load(file)

        # add bbox to image
        annot_img = self.mark_annotations(img=img, annotations=annot_data["annotations"])

        return img, annot_img, mask_img, annot_data

    def mark_annotations(self, img, annotations):

        # to visualize results
        annoted_img = img.copy()
        for component in annotations:
            ratio = component["degradation"]
            bbox = component["bounding_box"]
            idx = component["id"]
            if ratio < 0:
                continue
            bbox = hp.box_coco_to_corner(bbox)
            annoted_img = hp.add_bbox(img=annoted_img, bbox=bbox, label=f"{idx}:{round(ratio, 3)}", font_scale=0.9)

        return annoted_img


