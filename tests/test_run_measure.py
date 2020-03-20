import os

# dummy tests
os.system("python ./citlab_article_separation_measure/run_measure.py --path_to_gt_xml_lst {} --path_to_hy_xml_lst {}".
          format("./tests/resources/dummy_examples/xml_paths_gt.lst",
                 "./tests/resources/dummy_examples/xml_paths_hy.lst"))

print("#" * 125)
print("#" * 125)
print("#" * 125)

# real tests
os.system("python ./citlab_article_separation_measure/run_measure.py --path_to_gt_xml_lst {} --path_to_hy_xml_lst {}".
          format("./tests/resources/real_examples/xml_paths_gt.lst",
                 "./tests/resources/real_examples/xml_paths_hy.lst"))
