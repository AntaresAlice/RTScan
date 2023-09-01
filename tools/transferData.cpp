#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

typedef uint32_t CODE;
#define CODEWIDTH 32

int main() {

    ifstream inputfile;

    inputfile.open("dateMap.txt");

    if(!inputfile.is_open()) {
        cout << "inputfile not found!" << endl;
        exit(0);
    }
    vector<int> inputData;
    while(inputfile) {
        int temp;
        inputfile >> temp;
        inputData.push_back(temp);
    }

    int N = inputData.size();

    CODE *data = (CODE *)malloc(N * sizeof(CODE));

    for (int i = 0; i < inputData.size(); i++) {
        data[i] = inputData[i];
    }

    //save data
    char DATA_PATH2[256] = "./dataForCBindex\0";
    FILE *fp2;
    if (!(fp2 = fopen(DATA_PATH2, "wb"))) {
      printf("save_to_file: fopen(%s) faild\n", DATA_PATH2);
      exit(-1);
    }
    printf("saving data to %s\n", DATA_PATH2);

    // 8/16/32 only
    if (CODEWIDTH == 8) {
      if (fwrite(data, sizeof(uint8_t), N, fp2) == 0) {
        printf("save_to_file: fwrite faild.\n");
        exit(-1);
      }
    } else if (CODEWIDTH == 16) {
      if (fwrite(data, sizeof(uint16_t), N, fp2) == 0) {
        printf("save_to_file: fwrite faild.\n");
        exit(-1);
      }
    } else if (CODEWIDTH == 32) {
      if (fwrite(data, sizeof(uint32_t), N, fp2) == 0) {
        printf("save_to_file: fwrite faild.\n");
        exit(-1);
      }
    } else {
      printf("CODE_WIDTH %d\n",CODEWIDTH);
      printf("save_to_file: CODE_WIDTH != 8/16/32.\n");
    }
    fclose(fp2);

    printf("[+] save finished!\n");
}