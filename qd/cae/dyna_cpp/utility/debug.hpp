
#ifndef DEBUG_HPP_
#define DEBUG_HPP_

namespace qd {

template<class D>
struct traced
{

#ifdef QD_DEBUG

public:
  traced() = default;
  traced(traced const&) { std::cout << typeid(D).name() << " copied\n"; }
  virtual ~traced() { std::cout << typeid(D).name() << " deleted\n"; };

protected:
  //   ~traced() = default;

#endif // ifdef:QD_DEBUG
};

} // namespace:qd

#endif