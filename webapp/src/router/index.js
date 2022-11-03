import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import AboutView from '../views/AboutView.vue'
import FacesView from '../views/FacesView.vue'
import TransitionsView from '../views/TransitionsView.vue'
import GenerateFromImageView from '../views/GenerateFromImageView.vue'
import FeaturesView from '../views/FeaturesView.vue'
import SearchView from '../views/SearchView.vue'
import InterchangeView from '../views/InterchangeView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  {
    path: '/faces',
    name: 'faces',
    component: FacesView
  },
  {
    path: '/search',
    name: 'search',
    component: SearchView
  },
  {
    path: '/transitions',
    name: 'transitions',
    component: TransitionsView
  },
  {
    path: '/fromImage',
    name: 'fromImage',
    component: GenerateFromImageView
  },
  {
    path: '/features',
    name: 'features',
    component: FeaturesView
  },
  {
    path: '/interchange',
    name: 'interchange',
    component: InterchangeView
  },
  {
    path: '/about',
    name: 'about',
    component: AboutView
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
