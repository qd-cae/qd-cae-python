
#ifndef DEBUG_HPP_
#define DEBUG_HPP_

#ifdef QD_DEBUG

namespace qd {

template<class D>
struct traced
{
public:
    traced() = default;
    traced(traced const&) { std::cout << typeid(D).name() << " copied\n"; }

protected:
    ~traced() = default;
};

} // namespace:qd

#endif // ifdef:QD_DEBUG
#endif