// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of your cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-bookshelf",
          title: "bookshelf",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/books/";
          },
        },{id: "post-in-defense-of-communication-technology",
        
          title: 'In Defense of Communication Technology <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "Due to the COVID-19 pandemic, everything from business meetings to college classes has moved to a digital format. With the rapid increase in the use of Zoom and other forms of digital communication, the effect of communication technology on human relationships has been questioned.",
        section: "Posts",
        handler: () => {
          
            window.open("https://www.linkedin.com/pulse/defense-communication-technology-sanjana-nandi/", "_blank");
          
        },
      },{id: "books-the-alchemist",
          title: 'The Alchemist',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_alchemist/";
            },},{id: "projects-exploring-reward-sharing-strategies-for-effective-cooperative-multi-agent-task-completion",
          title: 'Exploring Reward Sharing Strategies for Effective Cooperative Multi-Agent Task Completion',
          description: "Final project for CS 5756 Introduction to Robot Learning at Cornell University (M.Eng. CS program)",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cs5756-final-proj/";
            },},{id: "projects-adjusting-daytime-dehazing-algorithms-to-low-light-and-nighttime-scenarios",
          title: 'Adjusting Daytime Dehazing Algorithms to Low-Light and Nighttime Scenarios',
          description: "Final project for CS 6662 Computational Imaging at Cornell University (M.Eng. CS program)",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cs6662-final-proj/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%73%61%6E%6A%61%6E%61@%6E%61%6E%64%69.%6E%65%74", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/Sanjana679", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/sanjananandi", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
