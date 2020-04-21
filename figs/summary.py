import os
import pptx

from glob import glob

sdirs    = ['inverse_crime','50','8']
f1_files = []

for sdir in sdirs:
  # find all the files without noise and no prior
  f1_files += sorted(glob(os.path.join(sdir,'*beta_0_*noise_level_0.0_*f1.png')))
  
  # add files with noise and no prior
  f1_files += sorted(glob(os.path.join(sdir,'*beta_0_*noise_level_0.05_*f1.png')))
  
  # add prior recons with no noise
  f1_files += sorted(glob(os.path.join(sdir,'*beta_0.1_*noise_level_0.0_*f1.png')))
  f1_files += sorted(glob(os.path.join(sdir,'*beta_1.0_*noise_level_0.0_*f1.png')))
  f1_files += sorted(glob(os.path.join(sdir,'*beta_10.0_*noise_level_0.0_*f1.png')))
  
  # add prior recons with noise
  f1_files += sorted(glob(os.path.join(sdir,'*beta_0.1_*noise_level_0.05_*f1.png')))
  f1_files += sorted(glob(os.path.join(sdir,'*beta_1.0_*noise_level_0.05_*f1.png')))
  f1_files += sorted(glob(os.path.join(sdir,'*beta_10.0_*noise_level_0.05_*f1.png')))

f2_files = [x.replace('f1.png','f2.png') for x in f1_files]

prs = pptx.Presentation()
prs.slide_width  = pptx.util.Inches(8)
prs.slide_height = pptx.util.Inches(4.5)

for i, f1_file in enumerate(f1_files):
  blank_slide_layout = prs.slide_layouts[6]
  slide = prs.slides.add_slide(blank_slide_layout)
  
  pic = slide.shapes.add_picture(f1_file, pptx.util.Inches(0), pptx.util.Inches(0.75),            
                                 height = pptx.util.Inches(3))

  pic = slide.shapes.add_picture(f2_files[i], pptx.util.Inches(4), pptx.util.Inches(0.75),            
                                 height = pptx.util.Inches(3))
  

prs.save("summary.pptx")
