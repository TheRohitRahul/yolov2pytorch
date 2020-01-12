import xml.etree.ElementTree as ET 
import cv2
import os

debug_image_path = "./debug_images"
if not(os.path.exists(debug_image_path)):
  os.makedirs(debug_image_path)

all_classes = []

def plot_on_image(annotation_path, image_path):
  annotations = parse_xml(annotation_path)
  image = cv2.imread(image_path)
  for annotation in annotations:
    x1, y1, x2, y2, classname = annotation
    all_classes.append(classname)
    cv2.rectangle(image, (x1, y1), (x2,y2), (0,0,255), 4)
  cv2.imwrite(os.path.join(debug_image_path, os.path.basename(image_path)), image)


def parse_xml(xmlfile):
  tree = ET.parse(xmlfile)
  root = tree.getroot()
  annotations = []
  for item in root.findall("./object"):
    single_annot = [None]*5
    for child in item:
      if child.tag == "bndbox":
        for child_of_child in child:      
          if child_of_child.tag == "xmin":
            single_annot[0] = int(float(child_of_child.text))
          elif child_of_child.tag == "ymin":
            single_annot[1] = int(float(child_of_child.text))
          elif child_of_child.tag == "xmax":
            single_annot[2] = int(float(child_of_child.text))
          elif child_of_child.tag == "ymax":
            single_annot[3] = int(float(child_of_child.text))
      if child.tag == "name":
        single_annot[4] = child.text
    annotations.append(single_annot)
  return annotations


def parse_entire_folder(image_folder, annotation_folder):
  all_image_names = os.listdir(image_folder)
  all_annotation_names = os.listdir(annotation_folder)
  annotation_files = []
  image_files = []

  for image_name in all_image_names:
    if image_name.endswith(".jpg") or image_name.endswith(".png"):
      image_files.append(os.path.join(image_folder, image_name))
  
  for annotation_name in all_annotation_names:
    if annotation_name.endswith(".xml"):
      annotation_files.append(os.path.join(annotation_folder, annotation_name))

  for image_path in image_files:
    annotation_path = os.path.join(annotation_folder, os.path.splitext(os.path.basename(image_path))[0] + ".xml")
    plot_on_image(annotation_path, image_path)

if __name__ == "__main__":
  pass


