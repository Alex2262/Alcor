
#ifndef ALCOR_MCTS_H
#define ALCOR_MCTS_H

#include <unordered_set>
#include "constants.h"
#include "position.h"

class Node {
public:
    uint32_t parent = 0;
    uint32_t children_start = 0;
    uint32_t children_end = 0;
    double win_count = 0;
    int visits = 0;
    Move last_move;
    double policy = 0.0;

    Node(uint32_t c_parent, Move c_last_move) {
        parent = c_parent;
        win_count = 0;
        visits = 0;
        last_move = c_last_move;
    }
};


class Tree {
public:
    std::vector<Node> graph{};
};

class MCTS {
public:
    MCTS() = default;

    bool stopped = true;

    PLY_TYPE max_depth = MAX_DEPTH;
    uint64_t max_time = MAX_TIME;
    uint64_t max_nodes = MAX_ITERATIONS;

    Position position{};

    uint64_t start_time = 0;
    PLY_TYPE seldepth = 0;
    double average_depth = 1;
    uint64_t nodes = 0;

    uint32_t root_node_index = 0;
    int ply = 0;

    PLY_TYPE fifty_move = 0;

    Tree tree{};

    std::unordered_set<HASH_TYPE> main_game_hashes{};
    std::unordered_set<HASH_TYPE> tree_hashes{};

    void new_game();
    void update_tree(Move move);
    void print_info();

    void descend_to_root(uint32_t node_index);
    bool detect_repetition(HASH_TYPE hash);

    double get_policy(uint32_t node_index);
    void set_children_policies(uint32_t node_index);
    uint32_t select_best_child(uint32_t node_index);
    uint32_t selection();
    void expansion(uint32_t node_index);

    double evaluate_mcts();
    void back_propagation(uint32_t node_index, double evaluation, int result);
    uint32_t get_best_node();
    void search();

    void flatten_tree();
};

#endif //ALCOR_MCTS_H
