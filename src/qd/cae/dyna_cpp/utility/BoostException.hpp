
#ifndef BOOSTEXCEPTION_HPP
#define BOOSTEXCEPTION_HPP

namespace boost {

#ifdef BOOST_NO_EXCEPTIONS

template<class E> inline void throw_exception(E const & e)
{
    throw e;
}

#endif // BOOST_NO_EXCEPTIONS

} // boost

#endif // BOOSTEXCEPTION_HPP
