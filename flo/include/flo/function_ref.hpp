#ifndef FLO_INCLUDED_FUNCTION_REF
#define FLO_INCLUDED_FUNCTION_REF

#include <type_traits>
#include <utility>

namespace nonstd
{
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B,T>::type;



template<typename Fn> class function_ref;
 
template<typename Ret, typename ...Args>
class function_ref<Ret(Args...)> final
{
  using void_ptr_t = void*;
  using callback_t = Ret (*)(void_ptr_t, Args...);
  callback_t m_callback = nullptr;
  void_ptr_t m_callable = nullptr;
  
  template<typename F>
  static constexpr Ret callback_fn(void_ptr_t i_ptr, Args... i_args) noexcept
  {
    return (*reinterpret_cast<F*>(i_ptr))(std::forward<Args>(i_args)...);
  }
 
public:
  constexpr function_ref() noexcept = default;
  constexpr function_ref(const function_ref&) noexcept = default;
  function_ref& operator=(const function_ref&) noexcept = default;
  constexpr function_ref(function_ref&&) noexcept = default;
  function_ref& operator=(function_ref&&) noexcept = default;
  ~function_ref() = default;

  constexpr function_ref(std::nullptr_t) noexcept {}

  template <typename Callable, 
           typename = enable_if_t<
             !std::is_same<remove_reference_t<Callable>, function_ref>::value>>
  constexpr function_ref(Callable&& i_callable) noexcept
      : m_callback(callback_fn<remove_reference_t<Callable>>),
        m_callable((void_ptr_t)std::addressof(i_callable)) 
 {}

  constexpr Ret operator()(Args&& ...i_args) noexcept
  {
    return m_callback(m_callable, std::forward<Args>(i_args)...);
  }

  constexpr operator bool() noexcept { return m_callback; }
};

}

#endif//FLO_INCLUDED_FUNCTION_REF
