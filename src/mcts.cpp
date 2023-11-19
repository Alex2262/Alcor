
#include <iostream>
#include <cmath>
#include <queue>
#include <cmath>
#include "mcts.h"
#include "see.h"


void MCTS::new_game() {
    tree.graph.clear();
    root_node_index = 0;
    fifty_move = 0;

    tree.graph.emplace_back(root_node_index, NO_MOVE);
}

void MCTS::update_tree(Move move) {
    for (int i = 0; i < tree.graph[root_node_index].children_end - tree.graph[root_node_index].children_start; i++) {
        uint32_t current_child_index = tree.graph[root_node_index].children_start + i;
        if (tree.graph[current_child_index].last_move == move) {
            root_node_index = current_child_index;
            return;
        }
    }

    tree.graph.emplace_back(root_node_index, move);
    root_node_index = tree.graph.size() - 1;
}

void MCTS::print_info() {

    std::string pv_line{};

    uint32_t original_root_node_index = root_node_index;

    std::vector<Move> attempted_moves{};

    while(true) {
        position.set_state(position.state_stack[ply], fifty_move);

        uint32_t best_node_index = get_best_node();
        if (tree.graph[best_node_index].visits < 2) break;

        pv_line += tree.graph[best_node_index].last_move.get_uci(position) + " ";

        root_node_index = best_node_index;
        if (tree.graph[root_node_index].children_end - tree.graph[root_node_index].children_start == 0) break;

        position.make_move(tree.graph[root_node_index].last_move, position.state_stack[ply], fifty_move);
        attempted_moves.push_back(tree.graph[root_node_index].last_move);

        ply++;
    }

    for (auto i = attempted_moves.size() - 1; i >= 0; i--) {
        ply--;
        position.undo_move(attempted_moves[i], position.state_stack[ply], fifty_move);
    }

    root_node_index = original_root_node_index;

    uint32_t best_node_index = get_best_node();

    auto eval = static_cast<double>(tree.graph[best_node_index].win_count) /
                static_cast<double>(tree.graph[best_node_index].visits);

    auto score = static_cast<int>(-std::log(1 / std::clamp<double>(eval, -1.0, 1.0) - 1) * SCALE);

    auto time = std::chrono::high_resolution_clock::now();
    uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::time_point_cast<std::chrono::milliseconds>(time).time_since_epoch()).count();

    auto elapsed_time = current_time - start_time;
    auto nps = static_cast<uint64_t>(static_cast<double>(nodes) /
                                     (static_cast<double>(elapsed_time) /
                                      1000.0));

    std::cout << "info nodes " << nodes
              << " depth " << std::lround(average_depth + 0.5)
              << " seldepth " << seldepth
              << " score cp " << score
              << " time " << elapsed_time
              << " nps " << nps
              << " pv " << pv_line
              << std::endl;
}


void MCTS::descend_to_root(uint32_t node_index) {
    while (node_index != root_node_index && ply > 0) {
        ply--;
        position.undo_move(tree.graph[node_index].last_move, position.state_stack[ply], fifty_move);
        node_index = tree.graph[node_index].parent;
    }
}

bool MCTS::detect_repetition(HASH_TYPE hash) {
    if (main_game_hashes.contains(hash) || tree_hashes.contains(hash)) return true;
    return false;
}

double MCTS::get_policy(uint32_t node_index) {
    double policy = 0.0;
    Node& child_node = tree.graph[node_index];

    Move last_move = child_node.last_move;
    Piece selected = position.board[last_move.origin()];
    Piece occupied = position.board[last_move.target()];

    auto selected_type = get_piece_type(selected, position.side);

    if (last_move.type() == MOVE_TYPE_PROMOTION) {
        if (last_move.promotion_type() == PROMOTION_QUEEN) policy += 4.0;
    }

    if (last_move.is_capture(position)) {

        auto occupied_type = get_piece_type(occupied, ~position.side);

        policy += (MVV_LVA_VALUES[occupied_type] - MVV_LVA_VALUES[selected_type]) / 1000.0;
        policy += 2.4;
    }

    if (get_static_exchange_evaluation(position, last_move, -108)) policy += 3;

    return std::exp(policy);
}

void MCTS::set_children_policies(uint32_t node_index) {
    Node& node = tree.graph[node_index];
    uint32_t n_children = node.children_end - node.children_start;

    double policy_sum = 0;
    for (int i = 0; i < n_children; i++) {
        uint32_t child_node_index = tree.graph[node_index].children_start + i;
        Node& child_node = tree.graph[child_node_index];

        child_node.policy = get_policy(child_node_index);
        policy_sum += child_node.policy;
    }

    for (int i = 0; i < n_children; i++) {
        uint32_t child_node_index = tree.graph[node_index].children_start + i;
        Node &child_node = tree.graph[child_node_index];

        child_node.policy /= policy_sum;
    }
}

uint32_t MCTS::select_best_child(uint32_t node_index) {
    Node& node = tree.graph[node_index];

    uint32_t n_children = node.children_end - node.children_start;

    uint32_t best_node_index = 0;
    double best_puct = -1000000;

    for (int i = 0; i < n_children; i++) {
        uint32_t child_node_index = tree.graph[node_index].children_start + i;
        Node& child_node = tree.graph[child_node_index];

        double exploration_score = EXPLORATION_CONSTANT * std::sqrt(static_cast<double>(node.visits));
        double prior_score = child_node.policy * (exploration_score / (static_cast<double>(1 + child_node.visits)));
        double value_score = child_node.visits == 0 ? 0 :
                static_cast<double>(child_node.win_count) /
                static_cast<double>(child_node.visits);

        // std::cout << child_node.win_count << " " << child_node.visits << std::endl;

        double puct = child_node.visits == 0 ?
                exploration_score * child_node.policy + 0.5 :  // FPU
                prior_score + value_score;

        if (puct > best_puct) {
            best_puct = puct;
            best_node_index = child_node_index;
        }
    }

    return best_node_index;
}

uint32_t MCTS::selection() {
    uint32_t leaf_node_index = root_node_index;

    tree_hashes.clear();

    PLY_TYPE depth = 0;
    while (true) {

        uint32_t n_children = tree.graph[leaf_node_index].children_end - tree.graph[leaf_node_index].children_start;

        if (n_children == 0) break;

        tree_hashes.insert(position.hash_key);

        leaf_node_index = select_best_child(leaf_node_index);

        position.set_state(position.state_stack[ply], fifty_move);
        position.make_move(tree.graph[leaf_node_index].last_move, position.state_stack[ply], fifty_move);

        depth++;
        ply++;

    }

    average_depth = (average_depth * (static_cast<double>(nodes) - 1.0) + depth) / static_cast<double>(nodes);

    seldepth = std::max<PLY_TYPE>(seldepth, depth);
    return leaf_node_index;
}

void MCTS::expansion(uint32_t node_index) {

    position.get_pseudo_legal_moves(position.scored_moves[ply]);
    position.set_state(position.state_stack[ply], fifty_move);
    tree.graph[node_index].children_start = tree.graph.size();

    for (ScoredMove scored_move : position.scored_moves[ply]) {

        bool attempt = position.make_move(scored_move.move, position.state_stack[ply], fifty_move);

        position.undo_move(scored_move.move, position.state_stack[ply], fifty_move);

        if (!attempt) continue;

        tree.graph.emplace_back(node_index, scored_move.move);
    }

    tree.graph[node_index].children_end = tree.graph.size();
}

double MCTS::evaluate_mcts() {
    return 1.0 / (1.0 + std::exp(-(position.nnue_state.evaluate(position.side) / CP_SCALE)));
}

void MCTS::back_propagation(uint32_t node_index, double evaluation, int result) {

    double p_result = result == DRAW_RESULT ? 0.5 :
                      result == NO_RESULT ? evaluation :
                      -1.0;
    // std::cout << p_result << std::endl;

    uint32_t current_node_index = node_index;
    while (true) {
        Node& current_node = tree.graph[current_node_index];

        p_result = 1 - p_result;

        current_node.visits++;
        current_node.win_count += p_result;

        if (current_node.parent == current_node_index) break;  // Hit root

        current_node_index = current_node.parent;
    }
}

uint32_t MCTS::get_best_node() {
    int best = -1;
    uint32_t best_index = 0;
    for (int i = 0; i < tree.graph[root_node_index].children_end - tree.graph[root_node_index].children_start; i++) {
        Node& node = tree.graph[tree.graph[root_node_index].children_start + i];
        if (node.visits >= best) {
            best = node.visits;
            best_index = tree.graph[root_node_index].children_start + i;
        }
    }

    return best_index;
}

void MCTS::search() {
    seldepth = 0;
    nodes = 0;
    average_depth = 0.0;
    seldepth = 0;
    stopped = false;

    flatten_tree();

    auto time = std::chrono::high_resolution_clock::now();
    start_time = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::time_point_cast<std::chrono::milliseconds>(time).time_since_epoch()).count();

    uint32_t selected_node_index = root_node_index;

    for (nodes = 1; nodes <= max_nodes; nodes++) {

        descend_to_root(selected_node_index);
        selected_node_index = selection();

        int node_result = NO_RESULT;

        if (tree.graph[selected_node_index].visits > 0) {

            if (selected_node_index != root_node_index &&
                (detect_repetition(position.hash_key) || fifty_move == 100)) node_result = DRAW_RESULT;
            else {
                tree_hashes.insert(position.hash_key);
                expansion(selected_node_index);
                set_children_policies(selected_node_index);

                if (tree.graph[selected_node_index].children_end > tree.graph[selected_node_index].children_start) {
                    position.set_state(position.state_stack[ply], fifty_move);

                    int random_index = rand() % (tree.graph[selected_node_index].children_end - tree.graph[selected_node_index].children_start);
                    selected_node_index = tree.graph[selected_node_index].children_start + random_index;

                    position.make_move(tree.graph[selected_node_index].last_move, position.state_stack[ply], fifty_move);
                    ply++;

                    if (detect_repetition(position.hash_key)) node_result = DRAW_RESULT;

                } else {
                    node_result = position.is_attacked(position.get_king_pos(position.side), position.side) ?
                                  position.side ^ 1 : DRAW_RESULT;
                }
            }
        }

        double evaluation = node_result == NO_RESULT ? evaluate_mcts() : 0;

        // std::cout << "back propagation" << std::endl;
        back_propagation(selected_node_index, evaluation, node_result);

        //

        if ((nodes % 2048) == 0) {
            auto time = std::chrono::high_resolution_clock::now();
            uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::time_point_cast<std::chrono::milliseconds>(time).time_since_epoch()).count();

            if (current_time - start_time >= max_time) {
                break;
            }
        }

        if ((nodes % 30000) == 0) {
            descend_to_root(selected_node_index);
            print_info();
        }

        if (std::lround(average_depth + 0.5) >= max_depth) break;
        if (stopped) break;
    }

    descend_to_root(selected_node_index);
    print_info();
    std::cout << "bestmove " << tree.graph[get_best_node()].last_move.get_uci(position) << std::endl;
}

void MCTS::flatten_tree() {
    std::vector copy_graph = tree.graph;

    auto start_size = tree.graph.size();

    tree.graph.clear();

    std::queue<std::pair<uint32_t, uint32_t>> next_nodes_index;
    next_nodes_index.push({root_node_index, 0});
    tree.graph.push_back(copy_graph[root_node_index]);

    root_node_index = 0;

    tree.graph[0].parent = root_node_index;

    while (!next_nodes_index.empty()) {
        auto current_node_index = next_nodes_index.front();
        uint32_t old_node_index = current_node_index.first;
        uint32_t new_node_index = current_node_index.second;

        Node& current_old_node = copy_graph[old_node_index];
        Node& current_new_node = tree.graph[new_node_index];

        current_new_node.children_start = tree.graph.size();
        for (int i = 0; i < current_old_node.children_end - current_old_node.children_start; i++) {
            tree.graph.push_back(copy_graph[current_old_node.children_start + i]);
            tree.graph[tree.graph.size() - 1].parent = new_node_index;

            next_nodes_index.push({current_old_node.children_start + i, tree.graph.size() - 1});
        }

        current_new_node.children_end = tree.graph.size();

        next_nodes_index.pop();
    }

    std::cout << "Tree flattened from " << start_size << " to " << tree.graph.size() << std::endl;
}


