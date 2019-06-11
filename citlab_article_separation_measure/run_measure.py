# -*- coding: utf-8 -*-

import jpype
import datetime
import numpy as np
from argparse import ArgumentParser

from citlab_article_separation.io import get_article_polys_from_file
from citlab_python_util.io.file_loader import load_text_file
from citlab_python_util.math.measure import f_measure

from citlab_article_separation_measure.eval_measure import BaselineMeasureEval


def greedy_alignment(array):
    assert type(array) == np.ndarray, "array has to be np.ndarray"
    assert len(array.shape) == 2, "array has to be 2d matrix"
    assert array.dtype == float, "array has to be float"

    arr = np.copy(array)
    greedy_alignment = []  # np.zeros([min(*matrix.shape)])

    while True:
        # calculate indices for maximum alignment
        max_idx_x, max_idx_y = np.unravel_index(np.argmax(arr), arr.shape)
        # finish if all elements have been aligned
        if arr[max_idx_x, max_idx_y] < 0:
            break
        # get max alignment
        greedy_alignment.append((max_idx_x, max_idx_y))
        # set row and column to -1
        arr[max_idx_x, :] = -1.0
        arr[:, max_idx_y] = -1.0

    return greedy_alignment


def sum_over_indices(array, index_list):
    assert type(array) == np.ndarray, "array has to be np.ndarray"
    assert type(index_list) == list, "index_list has to be list"
    assert all([type(ele) == list or type(ele) == tuple for ele in index_list]),\
        "elements of index_list have to tuples or lists"

    index_size = len(index_list[0])
    assert all([len(ele) == index_size for ele in index_list]), "indices in index_list have to be of same length"
    assert len(array.shape) == index_size, "array shape and indices have to match"

    res = 0.0
    for index in index_list:
        res += array[index]

    return res


def run_eval(truth_file, reco_file, min_tol, max_tol, threshold_tf, java_code=True):
    """

    :param truth_file:
    :param reco_file:
    :param min_tol:
    :param max_tol:
    :param threshold_tf:
    :param java_code: usage of methods written in java or not
    """
    # # store the whole console output in a file
    # sys.stdout = open('results', 'w')

    if not (truth_file and reco_file):
        print("No arguments given for <truth> or <reco>, exiting. See --help for usage.")
        exit(1)

    # Parse input
    list_truth = []
    list_reco = []

    if truth_file.endswith((".txt", ".xml")):
        list_truth.append(truth_file)
    if reco_file.endswith((".txt", ".xml")):
        list_reco.append(reco_file)
    if truth_file.endswith(".lst") and reco_file.endswith(".lst"):
        try:
            list_truth = load_text_file(truth_file)
            list_reco = load_text_file(reco_file)
        except IOError:
            raise IOError("Cannot open truth- and/or reco-file.")

    if not (list_truth and list_reco):
        raise ValueError("Truth- and/or reco-file empty.")
    if not (len(list_truth) == len(list_reco)):
        raise ValueError("Same reco- and truth-list length required.")

    print("-----Article Segmentation Evaluation-----\n")
    print("Evaluation performed on {}".format(datetime.datetime.now().strftime("%Y.%m.%d, %H:%M")))
    print("Evaluation performed for GT  : {}".format(truth_file))
    print("Evaluation performed for HYPO: {}".format(reco_file))
    print("Number of pages: {}".format(len(list_truth)) + "\n")
    print("Loading protocol:")

    pages_article_truth_without_none = []
    pages_article_reco_without_none = []

    pages_article_truth_with_none = []
    pages_article_reco_with_none = []

    num_article_truth, num_poly_truth_without_none, num_poly_truth_with_none = 0, 0, 0
    num_article_reco, num_poly_reco_without_none, num_poly_reco_with_none = 0, 0, 0

    list_truth_fixed = list_truth[:]
    list_reco_fixed = list_reco[:]

    for i in range(len(list_truth)):
        truth_article_polys_from_file_without_none = None
        reco_article_polys_from_file_without_none = None

        truth_article_polys_from_file_with_none = None
        reco_article_polys_from_file_with_none = None

        # Get truth polygons article wise
        try:
            truth_article_polys_from_file_without_none, truth_article_polys_from_file_with_none, error_truth \
                = get_article_polys_from_file(list_truth[i])
        except IOError:
            error_truth = True

        # Get reco polygons article wise
        try:
            reco_article_polys_from_file_without_none, reco_article_polys_from_file_with_none, error_reco \
                = get_article_polys_from_file(list_reco[i])
        except IOError:
            error_reco = True

        # Skip pages with errors in either truth or reco
        if not (error_truth or error_reco):
            if truth_article_polys_from_file_without_none is not None and \
                    reco_article_polys_from_file_without_none is not None:

                pages_article_truth_without_none.append(truth_article_polys_from_file_without_none)
                pages_article_reco_without_none.append(reco_article_polys_from_file_without_none)

                pages_article_truth_with_none.append(truth_article_polys_from_file_with_none)
                pages_article_reco_with_none.append(reco_article_polys_from_file_with_none)

                # Count articles and polygons
                num_article_truth += len(truth_article_polys_from_file_without_none)
                num_poly_truth_without_none += sum(len(polys) for polys in truth_article_polys_from_file_without_none)
                num_poly_truth_with_none += sum(len(polys) for polys in truth_article_polys_from_file_with_none)

                num_article_reco += len(reco_article_polys_from_file_without_none)
                num_poly_reco_without_none += sum(len(polys) for polys in reco_article_polys_from_file_without_none)
                num_poly_reco_with_none += sum(len(polys) for polys in reco_article_polys_from_file_with_none)

        else:
            if truth_article_polys_from_file_with_none is not None and \
                    reco_article_polys_from_file_with_none is not None:

                if error_truth:
                    print("Warning loading: {}, only bd measure will be evaluated, since only \"None\" "
                          "baselines in GT.".format(list_truth[i]))
                if error_reco:
                    print("Warning loading: {}, only bd measure will be evaluated, since only \"None\" "
                          "baselines in HYPO.".format(list_reco[i]))

                pages_article_truth_without_none.append(None)
                pages_article_reco_without_none.append(None)

                pages_article_truth_with_none.append(truth_article_polys_from_file_with_none)
                pages_article_reco_with_none.append(reco_article_polys_from_file_with_none)

                # Count polygons
                num_poly_truth_with_none += sum(len(polys) for polys in truth_article_polys_from_file_with_none)
                num_poly_reco_with_none += sum(len(polys) for polys in reco_article_polys_from_file_with_none)
            else:
                if error_truth:
                    print("Error loading: {}, skipped.".format(list_truth[i]))
                if error_reco:
                    print("Error loading: {}, skipped.".format(list_reco[i]))
                list_truth_fixed.remove(list_truth[i])
                list_reco_fixed.remove(list_reco[i])

    if len(list_truth) == len(list_truth_fixed):
        print("Everything loaded without errors.")

    print("\n{} out of {} GT-HYPO page pairs loaded without errors and used for evaluation.".
          format(len(list_truth_fixed), len(list_truth)))
    print("Number of GT  : {} lines found in {} articles (exclusive \"None\" class)".
          format(num_poly_truth_without_none, num_article_truth))
    print("Number of HYPO: {} lines found in {} articles (exclusive \"None\" class)\n".
          format(num_poly_reco_without_none, num_article_reco))

    print("Number of GT  : {} lines found (inclusive \"None\" class)".format(num_poly_truth_with_none))
    print("Number of HYPO: {} lines found (inclusive \"None\" class)".format(num_poly_reco_with_none))

    #####################
    # Pagewise Evaluation
    print("\n-----Pagewise Evaluation-----\n")
    print("{:>16s} {:>10s} {:>10s} {:>10s}  {:^50s}  {:^50s}".
          format("Mode", "P-value", "R-value", "F-value", "TruthFile", "HypoFile"))
    print("-" * (15 + 1 + 10 + 1 + 10 + 1 + 10 + 1 + 50 + 1 + 50))

    as_recall_sum, as_precision_sum, as_f_measure_sum = 0, 0, 0
    as_recall_weighted_sum, as_precision_weighted_sum, as_f_measure_weighted_sum = 0, 0, 0

    bd_recall_without_none_sum, bd_precision_without_none_sum, bd_f_measure_without_none_sum = 0, 0, 0
    bd_recall_with_none_sum, bd_precision_with_none_sum, bd_f_measure_with_none_sum = 0, 0, 0

    with_none_counter, without_none_counter = 0, 0

    for page_index, page_articles in enumerate(zip(pages_article_truth_without_none, pages_article_reco_without_none)):
        page_articles_truth = page_articles[0]
        page_articles_reco = page_articles[1]

        if page_articles_truth is None and page_articles_reco is None:
            with_none_counter += 1

            # Create baseline measure evaluation
            bl_measure_eval = BaselineMeasureEval(min_tol, max_tol)

            # Evaluate baseline detection measure for entire baseline set with "None" class
            page_truth_with_none = [poly_truth for article_truth in pages_article_truth_with_none[page_index]
                                    for poly_truth in article_truth]
            page_reco_with_none = [poly_reco for article_reco in pages_article_reco_with_none[page_index]
                                   for poly_reco in article_reco]

            bl_measure_eval.calc_measure_for_page_baseline_polys(page_truth_with_none, page_reco_with_none,
                                                                 use_java_code=java_code)
            # R value
            page_recall_with_none = bl_measure_eval.measure.result.page_wise_recall[-1]
            bd_recall_with_none_sum += page_recall_with_none
            # P value
            page_precision_with_none = bl_measure_eval.measure.result.page_wise_precision[-1]
            bd_precision_with_none_sum += page_precision_with_none
            # F value
            page_f_measure_with_none = f_measure(page_precision_with_none, page_recall_with_none)
            bd_f_measure_with_none_sum += page_f_measure_with_none

            # Output
            print("{:>16s} {:>10s} {:>10s} {:>10s}  {}  {}".
                  format("as ind", "-", "-", "-", list_truth_fixed[page_index], list_reco_fixed[page_index]))
            print("{:>16s} {:>10s} {:>10s} {:>10s}  {}  {}".
                  format("as weighted, ind", "-", "-", "-", list_truth_fixed[page_index], list_reco_fixed[page_index]))
            print("{:>16s} {:>10s} {:>10s} {:>10s}  {}  {}".
                  format("bd without None", "-", "-", "-", list_truth_fixed[page_index], list_reco_fixed[page_index]))
            print("{:>16s} {:>10.4f} {:>10.4f} {:>10.4f}  {}  {}".
                  format("bd with None", page_precision_with_none, page_recall_with_none, page_f_measure_with_none,
                         list_truth_fixed[page_index], list_reco_fixed[page_index]) + "\n")

        else:
            without_none_counter += 1

            # Create precision & recall matrices for article wise comparisons
            page_wise_article_precision = np.zeros([len(page_articles_truth), len(page_articles_reco)])
            page_wise_article_recall = np.zeros([len(page_articles_truth), len(page_articles_reco)])

            # Create baseline measure evaluation
            bl_measure_eval = BaselineMeasureEval(min_tol, max_tol)

            # Evaluate measure for each article
            for i, article_truth in enumerate(page_articles_truth):
                for j, article_reco in enumerate(page_articles_reco):
                    bl_measure_eval.calc_measure_for_page_baseline_polys(article_truth, article_reco,
                                                                         use_java_code=java_code)
                    page_wise_article_precision[i, j] = bl_measure_eval.measure.result.page_wise_precision[-1]
                    page_wise_article_recall[i, j] = bl_measure_eval.measure.result.page_wise_recall[-1]

            # Evaluate baseline detection measure for entire baseline set without "None" class
            page_truth_without_none = [poly_truth for article_truth in page_articles_truth for poly_truth in article_truth]
            page_reco_without_none = [poly_reco for article_reco in page_articles_reco for poly_reco in article_reco]

            bl_measure_eval.calc_measure_for_page_baseline_polys(page_truth_without_none, page_reco_without_none,
                                                                 use_java_code=java_code)
            # R value
            page_recall_without_none = bl_measure_eval.measure.result.page_wise_recall[-1]
            bd_recall_without_none_sum += page_recall_without_none
            # P value
            page_precision_without_none = bl_measure_eval.measure.result.page_wise_precision[-1]
            bd_precision_without_none_sum += page_precision_without_none
            # F value
            page_f_measure_without_none = f_measure(page_precision_without_none, page_recall_without_none)
            bd_f_measure_without_none_sum += page_f_measure_without_none

            # Evaluate baseline detection measure for entire baseline set with "None" class
            page_truth_with_none = [poly_truth for article_truth in pages_article_truth_with_none[page_index]
                                    for poly_truth in article_truth]
            page_reco_with_none = [poly_reco for article_reco in pages_article_reco_with_none[page_index]
                                   for poly_reco in article_reco]

            bl_measure_eval.calc_measure_for_page_baseline_polys(page_truth_with_none, page_reco_with_none,
                                                                 use_java_code=java_code)
            # R value
            page_recall_with_none = bl_measure_eval.measure.result.page_wise_recall[-1]
            bd_recall_with_none_sum += page_recall_with_none
            # P value
            page_precision_with_none = bl_measure_eval.measure.result.page_wise_precision[-1]
            bd_precision_with_none_sum += page_precision_with_none
            # F value
            page_f_measure_with_none = f_measure(page_precision_with_none, page_recall_with_none)
            bd_f_measure_with_none_sum += page_f_measure_with_none

            # Greedy alignment of articles
            #####
            # 1) Without article weighting; independent alignment
            greedy_align_precision = greedy_alignment(page_wise_article_precision)
            greedy_align_recall = greedy_alignment(page_wise_article_recall)
            # P value
            precision = sum_over_indices(page_wise_article_precision, greedy_align_precision)
            precision = precision / len(page_articles_reco)
            as_precision_sum += precision
            # R value
            recall = sum_over_indices(page_wise_article_recall, greedy_align_recall)
            recall = recall / len(page_articles_truth)
            as_recall_sum += recall
            # F value
            f_measure_all = f_measure(precision, recall)
            as_f_measure_sum += f_measure_all

            #####
            # 2) With article weighting (based on baseline percentage portion of truth/hypo); independent alignment
            articles_truth_length = np.asarray([len(l) for l in page_articles_truth], dtype=np.float32)
            articles_reco_length = np.asarray([len(l) for l in page_articles_reco], dtype=np.float32)
            articles_truth_weighting = articles_truth_length / np.sum(articles_truth_length)
            articles_reco_weighting = articles_reco_length / np.sum(articles_reco_length)

            # column-wise weighting for precision
            article_wise_precision_weighted = page_wise_article_precision * articles_reco_weighting

            # row-wise weighting for recall
            article_wise_recall_weighted = page_wise_article_recall * np.expand_dims(articles_truth_weighting, axis=1)

            greedy_align_precision_weighted = greedy_alignment(article_wise_precision_weighted)
            greedy_align_recall_weighted = greedy_alignment(article_wise_recall_weighted)
            # P value
            precision_weighted = sum_over_indices(article_wise_precision_weighted, greedy_align_precision_weighted)
            as_precision_weighted_sum += precision_weighted
            # R value
            recall_weighted = sum_over_indices(article_wise_recall_weighted, greedy_align_recall_weighted)
            as_recall_weighted_sum += recall_weighted
            # F value
            f_measure_weighted = f_measure(precision_weighted, recall_weighted)
            as_f_measure_weighted_sum += f_measure_weighted

            # Output
            print("{:>16s} {:>10.4f} {:>10.4f} {:>10.4f}  {}  {}".
                  format("as ind", precision, recall, f_measure_all,
                         list_truth_fixed[page_index], list_reco_fixed[page_index]))
            print("{:>16s} {:>10.4f} {:>10.4f} {:>10.4f}  {}  {}".
                  format("as weighted, ind", precision_weighted, recall_weighted, f_measure_weighted,
                         list_truth_fixed[page_index], list_reco_fixed[page_index]))
            print("{:>16s} {:>10.4f} {:>10.4f} {:>10.4f}  {}  {}".
                  format("bd without None", page_precision_without_none, page_recall_without_none,
                         page_f_measure_without_none, list_truth_fixed[page_index], list_reco_fixed[page_index]))
            print("{:>16s} {:>10.4f} {:>10.4f} {:>10.4f}  {}  {}".
                  format("bd with None", page_precision_with_none, page_recall_with_none, page_f_measure_with_none,
                         list_truth_fixed[page_index], list_reco_fixed[page_index]) + "\n")

    ##################
    # Final Evaluation
    print("\n-----Final Evaluation-----\n")

    if without_none_counter != 0:
        print("AS scores:\n")
        print("Average P-value  (ind): {:.4f}".format(as_precision_sum / without_none_counter) +
              "  Average P-value  (weighted, ind): {:.4f}".format(as_precision_weighted_sum / without_none_counter))
        print("Average R-value  (ind): {:.4f}".format(as_recall_sum / without_none_counter) +
              "  Average R-value  (weighted, ind): {:.4f}".format(as_recall_weighted_sum / without_none_counter))
        print("Average F1-score (ind): {:.4f}".format(as_f_measure_sum / without_none_counter) +
              "  Average F1-score (weighted, ind): {:.4f}".format(as_f_measure_weighted_sum / without_none_counter))

        print("\nBD scores:\n")
        print("Average P-value  (without \"None\"): {:.4f}".format(bd_precision_without_none_sum / without_none_counter) +
              "  Average P-value  (with \"None\"): {:.4f}".
              format(bd_precision_with_none_sum / (without_none_counter + with_none_counter)))
        print("Average R-value  (without \"None\"): {:.4f}".format(bd_recall_without_none_sum / without_none_counter) +
              "  Average R-value  (with \"None\"): {:.4f}".
              format(bd_recall_with_none_sum / (without_none_counter + with_none_counter)))
        print("Average F1-score (without \"None\"): {:.4f}".format(bd_f_measure_without_none_sum / without_none_counter) +
              "  Average F1-score (with \"None\"): {:.4f}".
              format(bd_f_measure_with_none_sum / (without_none_counter + with_none_counter)))
    else:
        print("AS scores:\n")
        print("Average P-value  (ind): {:>4s}".format("-") +
              "  Average P-value  (weighted, ind): {:>4s}".format("-"))
        print("Average R-value  (ind): {:>4s}".format("-") +
              "  Average R-value  (weighted, ind): {:>4s}".format("-"))
        print("Average F1-score (ind): {:>4s}".format("-") +
              "  Average F1-score (weighted, ind): {:>4s}".format("-"))

        print("\nBD scores:\n")
        print("Average P-value  (without \"None\"): {:>4s}".format("-") +
              "  Average P-value  (with \"None\"): {:.4f}".
              format(bd_precision_with_none_sum / (without_none_counter + with_none_counter)))
        print("Average R-value  (without \"None\"): {:>4s}".format("-") +
              "  Average R-value  (with \"None\"): {:.4f}".
              format(bd_recall_with_none_sum / (without_none_counter + with_none_counter)))
        print("Average F1-score (without \"None\"): {:>4s}".format("-") +
              "  Average F1-score (with \"None\"): {:.4f}".
              format(bd_f_measure_with_none_sum / (without_none_counter + with_none_counter)))

    # sys.stdout.close()


if __name__ == '__main__':
    # Argument parser and usage
    usage_string = """%(prog)s <truth> <reco> [OPTIONS]
    You can add specific options via '--OPTION VALUE'
    This method calculates the baseline errors in a precision/recall manner.
    As input it requires the truth and reco information.
    A basic truth (and reco) file corresponding to a page has to be a txt-file,
    where every line corresponds to a baseline polygon and should look like:
    x1,y1;x2,y2;x3,y3;...;xn,yn. Alternatively, the PageXml format is allowed.
    As arguments (truth, reco) such txt-files OR lst-files (containing a path to
    a basic txt-file per line) are required. For lst-files, the order of the
    truth/reco-files in both lists has to be identical."""
    parser = ArgumentParser(usage=usage_string)

    # Command-line arguments
    parser.add_argument('--truth', default='', type=str, metavar="STR",
                        help="truth-files in txt- or lst-format (see usage)")
    parser.add_argument('--reco', default='', type=str, metavar="STR",
                        help="reco-files in txt- or lst-format (see usage)")
    parser.add_argument('--min_tol', default=-1, type=int, metavar='FLOAT',
                        help="minimum tolerance value, -1 for dynamic calculation (default: %(default)s)")
    parser.add_argument('--max_tol', default=-1, type=int, metavar='FLOAT',
                        help="maximum tolerance value, -1 for dynamic calculation (default: %(default)s)")
    parser.add_argument('--threshold_tf', default=-1.0, type=float, metavar='FLOAT',
                        help="threshold for P- and R-value to make a decision concerning tp, fp, fn, tn. "
                             "Should be between 0 and 1 (default: %(default)s - nothing is done)")
    parser.add_argument('--java_code', default=True, type=bool, metavar='BOOL',
                        help="usage of methods written in java or not (default: %(default)s)")

    # def str2bool(arg):
    #     return arg.lower() in ('true', 't', '1')
    # parser.add_argument('--use_regions', default=False, nargs='?', const=True, type=str2bool, metavar='BOOL',
    #                     help="only evaluate hypo polygons if they are (partly) contained in region polygons,"
    #                          " if they are available (default: %(default)s)")

    # start java virtual machine to be able to execute the java code
    jpype.startJVM(jpype.getDefaultJVMPath())

    # example with Command-line arguments
    # flags = parser.parse_args()
    # run_eval(flags.truth, flags.reco, min_tol=flags.min_tol, max_tol=flags.max_tol, threshold_tf=flags.threshold_tf,
    #          java_code=flags.java_code)

    # # example with list of PageXml files
    gt_files_path_list = "./tests/resources/test_run_measure/gt_xml_paths.lst"
    hy_files_path_list = "./tests/resources/test_run_measure/hy_xml_paths.lst"

    # gt_files_path_list = "./test/resources/newseye_as_test_data/gt_xml_paths.lst"
    # hy_files_path_list = "./test/resources/newseye_as_test_data/hy_xml_paths.lst"

    # gt_files_path_list = "./test/resources/newseye_as_test_data_onb/gt_xml_paths.lst"
    # hy_files_path_list = "./test/resources/newseye_as_test_data_onb/hy_xml_paths.lst"

    # gt_files_path_list = "./test/resources/Le_Matin_Set/gt_xml_paths.lst"
    # hy_files_path_list = "./test/resources/Le_Matin_Set/hy_xml_paths.lst"

    run_eval(gt_files_path_list, hy_files_path_list, min_tol=-1, max_tol=-1, threshold_tf=-1, java_code=True)

    # example for the evaluation of one special page
    # newspaper_site = "19000715_1-0001.xml"
    # gt_files_path_list = "./test/resources/newseye_as_test_data/xml_files_gt/" + newspaper_site
    # hy_files_path_list = "./test/resources/newseye_as_test_data/xml_files_hy/" + newspaper_site
    # run_eval(gt_files_path_list, hy_files_path_list, min_tol=-1, max_tol=-1, threshold_tf=-1, java_code=True)

    # example with txt files (perfect bd)
    # gt_files_path_list = "./test/resources/perfect_bd_as_test_data/gt_txt_paths.lst"
    # hy_files_path_list = "./test/resources/perfect_bd_as_test_data/hy_txt_paths.lst"
    # run_eval(gt_files_path_list, hy_files_path_list, min_tol=-1, max_tol=-1, threshold_tf=-1, java_code=True)

    # example with txt files (imperfect bd)
    # gt_files_path_list = "./test/resources/imperfect_bd_as_test_data/gt_txt_paths.lst"
    # hy_files_path_list = "./test/resources/imperfect_bd_as_test_data/hy_txt_paths.lst"
    # run_eval(gt_files_path_list, hy_files_path_list, min_tol=-1, max_tol=-1, threshold_tf=-1, java_code=True)
    # # run_eval(gt_files_path_list, hy_files_path_list, min_tol=5, max_tol=25, threshold_tf=-1, java_code=True)

    # shut down the java virtual machine
    jpype.shutdownJVM()
