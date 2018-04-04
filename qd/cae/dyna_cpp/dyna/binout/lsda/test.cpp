
#include "LSDA_Wrapper.hpp"
#include <iostream>
#include <string>

/*
extern "C" {

#include "lsda.h"


int main(){
    
    std::cout << "\n" << "Opening Binout:" << std::endl;
    int handle = lsda_open("binout",LSDA_READONLY);
    std::cout << "lsda_getname: " << lsda_getname(handle) << std::endl;
    int nDir = 0;
    std::cout << "lsda_util_countdir: " << lsda_util_countdir(handle,"/",&nDir) << std::endl;
    std::cout << "lsda_util_countdir: " << lsda_readdir("/", childdirname, &tid, &len, &fno) << std::endl;
    lsda_close(handle);
    
    return 0;
}

}
*/

int main(){
    
    lasso::LSDA_Wrapper wrapper("G:\\Programming\\CPP\\Lasso-CAE-Analytics\\src\\io\\lsda\\binout");
    
    for(const auto& name : wrapper.scan_for_data()){
        std::cout << name << std::endl;
    }
    try {
        wrapper.read_secforc();
        wrapper.read_matsum();
    } catch (std::string ex) {
        std::cout << "Exception: " << ex << std::endl;
    }

    return 0;
}