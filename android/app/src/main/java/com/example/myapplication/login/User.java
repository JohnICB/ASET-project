public class User
{

    private String userId;
    private String userName;

    public LoggedInUser(String userId, String name)
    {
        this.userId = userId;
        this.userName = name;
    }

    public String getUserId()
    {
        return userId;
    }

    public String getUserName()
    {
        return userName;
    }
}