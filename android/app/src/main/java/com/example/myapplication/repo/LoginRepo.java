public class LoginRepo
{

    private static volatile LoginRepo instance;

    private LoginDataSource dataSource;
    private User user = null;

    // singleton
    private LoginRepo(LoginDataSource dataSource)
    {
        this.dataSource = dataSource;
    }

    public static LoginRepo getInstance(LoginDataSource dataSource)
    {
        if (instance == null)
        {
            instance = new LoginRepo(dataSource);
        }
        return instance;
    }

    public boolean isLoggedIn()
    {
        return user != null;
    }

    public void logout()
    {

        // TODO
        user = null;
        dataSource.logout();
    }

    private void setUser(User user)
    {
        this.user = user;
    }

    public Result<User> login(String username, String password)
    {
        // TODO handle login
    }
}