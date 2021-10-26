"""Training and testing unbiased learning to rank algorithms.

See the following paper for more information about different algorithms.
    
    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18
    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import ultra

# rank list size should be read from data
parser = argparse.ArgumentParser(description='Pipeline commandline argument')
parser.add_argument("--data_dir", type=str, default="./tests/data/", help="The directory of the experimental dataset.")
parser.add_argument("--train_data_prefix", type=str, default="train", help="The name prefix of the training data "
                                                                         "in data_dir.")
parser.add_argument("--valid_data_prefix", type=str, default="valid", help="The name prefix of the validation data in "
                                                                         "data_dir.")
parser.add_argument("--test_data_prefix", type=str, default="test", help="The name prefix of the test data in data_dir.")
parser.add_argument("--model_dir", type=str, default="./tests/tmp_model/", help="The directory for model and "
                                                                              "intermediate outputs.")
parser.add_argument("--output_dir", type=str, default="./tests/tmp_output/", help="The directory to output results.")

parser.add_argument("--click_model_dir", type=str, default=None, help="The directory that contains labels produced by the click model")
parser.add_argument("--data_format", type=str, default="ULTRA", help="The format of the data")
# model
parser.add_argument("--setting_file", type=str, default="./example/offline_setting/dla_exp_settings.json",
                    help="A json file that contains all the settings of the algorithm.")

# general training parameters
parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size to use during training.")
parser.add_argument("--max_list_cutoff", type=int, default=0,
                    help="The maximum number of top documents to consider in each rank list (0: no limit).")
parser.add_argument("--selection_bias_cutoff", type=int, default=10,
                    help="The maximum number of top documents to be shown to user "
                         "(which creates selection bias) in each rank list (0: no limit).")
parser.add_argument("--max_train_iteration", type=int, default=10000,
                    help="Limit on the iterations of training (0: no limit).")
parser.add_argument("--start_saving_iteration", type=int, default=0,
                    help="The minimum number of iterations before starting to test and save models. "
                         "(0: no limit).")
parser.add_argument("--steps_per_checkpoint", type=int, default=50,
                    help="How many training steps to do per checkpoint.")

parser.add_argument("--test_while_train", type=bool, default=False,
                    help="Set to True to test models during the training process.")
parser.add_argument("--test_only", type=bool, default=False,
                    help="Set to True for testing models only.")

args = parser.parse_args()


def create_model(exp_settings, data_set):
    """Create model and initialize or load parameters in session.
    
        Args:
            exp_settings: (dictionary) The dictionary containing the model settings.
            data_set: (Raw_data) The dataset used to build the input layer.
    """

    model = ultra.utils.find_class(exp_settings['learning_algorithm'])(data_set, exp_settings)
    try:
        checkpoint_path = os.path.join(args.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
        ckpt = torch.load(checkpoint_path)
        print("Reading model parameters from %s" % checkpoint_path)
        model.model.load_state_dict(ckpt)
        model.model.eval()
    except FileNotFoundError:
        print("Created model with fresh parameters.")
    return model


def train(exp_settings):
    # Prepare data.
    print("Reading data in %s" % args.data_dir)
    train_set = ultra.utils.read_data(args.data_dir, args.train_data_prefix, args.click_model_dir, args.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(train_set,
                                                                             exp_settings['train_input_hparams'],
                                                                             exp_settings)
    valid_set = ultra.utils.read_data(args.data_dir, args.valid_data_prefix, args.click_model_dir, args.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(valid_set,
                                                                             exp_settings['train_input_hparams'],
                                                                             exp_settings)

    print("Train Rank list size %d" % train_set.rank_list_size)
    print("Valid Rank list size %d" % valid_set.rank_list_size)
    exp_settings['max_candidate_num'] = max(train_set.rank_list_size, valid_set.rank_list_size)
    test_set = None
    if args.test_while_train:
        test_set = ultra.utils.read_data(args.data_dir, args.test_data_prefix, args.max_list_cutoff)
        ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set,
                                                                                 exp_settings['train_input_hparams'],
                                                                                 exp_settings)
        print("Test Rank list size %d" % test_set.rank_list_size)
        exp_settings['max_candidate_num'] = max(test_set.rank_list_size, exp_settings['max_candidate_num'])
        test_set.pad(exp_settings['max_candidate_num'])

    if 'selection_bias_cutoff' not in exp_settings:  # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = args.selection_bias_cutoff if args.selection_bias_cutoff > 0 else \
            exp_settings['max_candidate_num']

    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'],
                                                exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])

    # Pad data
    train_set.pad(exp_settings['max_candidate_num'])
    valid_set.pad(exp_settings['max_candidate_num'])

    # Create model based on the input layer.
    print("Creating model...")
    model = create_model(exp_settings, train_set)
    # model.print_info()

    # Create data feed
    train_input_feed = ultra.utils.find_class(exp_settings['train_input_feed'])(model, args.batch_size,
                                                                                exp_settings['train_input_hparams'])
    valid_input_feed = ultra.utils.find_class(exp_settings['valid_input_feed'])(model, args.batch_size,
                                                                                exp_settings['valid_input_hparams'])
    test_input_feed = None
    if args.test_while_train:
        test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, args.batch_size,
                                                                                  exp_settings[
                                                                                      'test_input_hparams'])

    # Create tensorboard summarizations.
    train_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/train_log')
    valid_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/valid_log')
    test_writer = None
    if args.test_while_train:
        test_writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.model_dir + '/test_log')

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    best_perf = None
    print("max_train_iter: ", args.max_train_iteration)
    while True:
        # Get a batch and make a step.
        start_time= time.time()
        input_feed, info_map = train_input_feed.get_batch(train_set, check_validation=True, data_format=args.data_format)
        step_loss, _, summary = model.train(input_feed)
        step_time += (time.time() - start_time) / args.steps_per_checkpoint
        loss += step_loss / args.steps_per_checkpoint
        current_step += 1
        train_writer.add_scalars("Training at step %s" % model.global_step, summary)

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % args.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            print("global step %d learning rate %.4f step-time %.2f loss "
                  "%.4f" % (model.global_step, model.learning_rate,
                            step_time, loss))
            previous_losses.append(loss)

            # Validate model
            def validate_model(data_set, data_input_feed):
                it = 0
                count_batch = 0.0
                summary_list = []
                batch_size_list = []
                while it < len(data_set.initial_list):
                    input_feed, info_map = data_input_feed.get_next_batch(
                        it, data_set, check_validation=False, data_format=args.data_format)
                    _, _, summary = model.validation(input_feed)
                    summary_list.append(summary)
                    batch_size_list.append(len(info_map['input_list']))
                    it += batch_size_list[-1]
                    count_batch += 1.0
                return ultra.utils.merge_Summary(summary_list, batch_size_list)
                # return summary_list

            valid_summary = validate_model(valid_set, valid_input_feed)
            # valid_writer.add_scalars('Validation Summary', valid_summary, model.global_step)
            for key,value in valid_summary.items():
                print(key, value)

            if args.test_while_train:
                test_summary = validate_model(test_set, test_input_feed)
                test_writer.add_scalars('Validation Summary while training', valid_summary, model.global_step)
                for key, value in test_summary.items:
                    print(key, value)

            # Save checkpoint if the objective metric on the validation set is better
            if "objective_metric" in exp_settings:
                for key,value in valid_summary.items():
                    if key == exp_settings["objective_metric"]:
                        if current_step >= args.start_saving_iteration:
                            if best_perf == None or best_perf < value:
                                checkpoint_path = os.path.join(args.model_dir,
                                                               "%s.ckpt" % exp_settings['learning_algorithm'])
                                torch.save(model.model.state_dict(), checkpoint_path)
                                best_perf = value
                                break
                        print('Save model, valid %s:%.3f' % (key, best_perf))

            # Save checkpoint if there is no objective metric
            if best_perf == None and current_step > args.start_saving_iteration:
                checkpoint_path = os.path.join(args.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                torch.save(model.model.state_dict(), checkpoint_path)
            if loss == float('inf'):
                break

            step_time, loss = 0.0, 0.0
            sys.stdout.flush()

            if args.max_train_iteration > 0 and current_step > args.max_train_iteration:
                print("current_step: ", current_step)
                break
    train_writer.close()
    valid_writer.close()
    if args.test_while_train:
        test_writer.close()


def test(exp_settings):
    # Load test data.
    print("Reading data in %s" % args.data_dir)
    test_set = ultra.utils.read_data(args.data_dir, args.test_data_prefix, args.click_model_dir, args.max_list_cutoff)
    ultra.utils.find_class(exp_settings['train_input_feed']).preprocess_data(test_set,
                                                                             exp_settings['train_input_hparams'],
                                                                             exp_settings)
    exp_settings['max_candidate_num'] = test_set.rank_list_size

    if 'selection_bias_cutoff' not in exp_settings:  # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = args.selection_bias_cutoff if args.selection_bias_cutoff > 0 else \
            exp_settings['max_candidate_num']
    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'],
                                                exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])
    
    test_set.pad(exp_settings['max_candidate_num'])

    # Create model and load parameters.
    model = create_model(exp_settings, test_set)

    # Create input feed
    test_input_feed = ultra.utils.find_class(exp_settings['test_input_feed'])(model, args.batch_size,
                                                                              exp_settings['test_input_hparams'])

    test_writer = SummaryWriter(log_dir = args.model_dir + '/test_log')

    rerank_scores = []
    summary_list = []
    # Start testing.

    it = 0
    count_batch = 0.0
    batch_size_list = []
    while it < len(test_set.initial_list):
        input_feed, info_map = test_input_feed.get_next_batch(it, test_set, check_validation=False)
        _, output_logits, summary = model.validation(input_feed)
        summary_list.append(summary)
        batch_size_list.append(len(info_map['input_list']))
        for x in range(batch_size_list[-1]):
            rerank_scores.append(output_logits[x])
        it += batch_size_list[-1]
        count_batch += 1.0
        print("Testing {:.0%} finished".format(float(it) / len(test_set.initial_list)), end="\r", flush=True)

    print("\n[Done]")
    test_summary = ultra.utils.merge_Summary(summary_list, batch_size_list)
    print("  eval: %s" % (
        ' '.join(['%s:%.3f' % (key, value) for key,value in test_summary.items()])
    ))

    # get rerank indexes with new scores
    rerank_lists = []
    for i in range(len(rerank_scores)):
        scores = rerank_scores[i]
        rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ultra.utils.output_ranklist(test_set, rerank_scores, args.output_dir, args.test_data_prefix)

    return


def main(_):
    exp_settings = json.load(open(args.setting_file))
    if args.test_only:
        test(exp_settings)
    else:
        train(exp_settings)


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
