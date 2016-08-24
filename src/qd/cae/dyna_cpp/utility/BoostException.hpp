
#ifndef BOOSTEXCEPTION_HPP
#define BOOSTEXCEPTION_HPP

namespace boost {

// WINDOWS
#ifdef _WIN32

template<class E> inline void throw_exception(E const & e)
{
    throw e;
}

// LINUX
#elif

#ifdef BOOST_NO_EXCEPTIONS
template<class E> inline void throw_exception(E const & e)
{
    throw e;
}
#endif

#endif

} // boost

#endif // BOOSTEXCEPTION_HPP
