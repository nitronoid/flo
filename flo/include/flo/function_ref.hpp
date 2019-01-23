#ifndef FLO_INCLUDED_FUNCTION_REF
#define FLO_INCLUDED_FUNCTION_REF

#include <type_traits>
#include <utility>

namespace nonstd
{

template<typename Fn> class function_ref;
 
template<typename Ret, typename ...Args>
class function_ref<Ret(Args...)> final
{
  using void_ptr_t = void const *;
  using callback_t = Ret (*)(void_ptr_t, Args...);
  callback_t m_callable = nullptr;
  void_ptr_t m_callable_ptr = nullptr;
  
  template <typename F>
  static constexpr auto make_erased_func() noexcept
  {
    return [](void_ptr_t i_ptr, Args... i_args) noexcept -> Ret
    {
        return (*reinterpret_cast<F*>(i_ptr))(std::forward<Args>(i_args)...);
    };
  }
 
public:
  constexpr function_ref() noexcept = default;
  constexpr function_ref(const function_ref&) noexcept = default;
  constexpr function_ref& operator=(const function_ref&) noexcept = default;
  constexpr function_ref(function_ref&&) noexcept = default;
  constexpr function_ref& operator=(function_ref&&) noexcept = default;
  ~function_ref() = default;

  constexpr function_ref(std::nullptr_t) noexcept {}

  template <typename Callable, 
           typename = std::enable_if_t<
             !std::is_same<std::remove_reference_t<Callable>, function_ref>::value>>
  constexpr function_ref(Callable&& i_callable) noexcept
      : m_callable(make_erased_func<std::remove_reference_t<Callable>>()),
        m_callable_ptr(reinterpret_cast<void_ptr_t>(&i_callable)) 
 {}

  constexpr Ret operator()(Args&& ...i_args) noexcept
  {
    return m_callable(m_callable_ptr, std::forward<Args>(i_args)...);
  }

  constexpr operator bool() noexcept { return m_callable; }
};

}

#endif//FLO_INCLUDED_FUNCTION_REF
