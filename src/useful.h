
#ifndef ALCOR_USEFUL_H
#define ALCOR_USEFUL_H

#include <vector>
#include <string>

template <typename Out>
void split(const std::string &s, char delim, Out result);

std::vector<std::string> split(const std::string &s, char delim);

Piece piece_to_num(char piece);

#endif //ALCOR_USEFUL_H