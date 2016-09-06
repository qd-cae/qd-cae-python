
#ifndef BOOSTEXCEPTION_HPP
#define BOOSTEXCEPTION_HPP

namespace boost {

// WINDOWS
#ifdef _WIN32

#ifdef BOOST_NO_EXCEPTIONS
template<class E> inline void throw_exception(E const & e)
{
    throw e;
}
#endif

// LINUX
#else

#ifdef BOOST_NO_EXCEPTIONS
template<class E> inline void throw_exception(E const & e)
{
    throw e;
}
#endif

#endif // _WIN32

} // boost

#endif // ifndef BOOSTEXCEPTION_HPP
